"""Command line: python -m tinker_extend.cli {ingest|query|stats|eval|ingest-wave|embed-backfill} ..."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinker_extend.config import load_config
from tinker_extend.embeddings import NullEmbedder, embedder_from_env
from tinker_extend.extend_client import ExtendClient
from tinker_extend.ingest import iter_candidate_files
from tinker_extend.pipeline import KnowledgeBasePipeline, WaveIngester, backfill_embeddings
from tinker_extend.qa import retrieve
from tinker_extend.store import KnowledgeBase


def build_pipeline(db_path: str | None) -> KnowledgeBasePipeline:
    cfg = load_config(kb_path=db_path)
    client = ExtendClient(
        api_key=cfg.api_key,
        api_version=cfg.api_version,
        base_url=cfg.base_url,
        workspace_id=cfg.workspace_id,
    )
    kb = KnowledgeBase(cfg.kb_path)
    return KnowledgeBasePipeline(
        client, kb,
        embedder=embedder_from_env(),
        poll_interval_s=cfg.poll_interval_s,
        poll_timeout_s=cfg.poll_timeout_s,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    pipe = build_pipeline(args.db)
    try:
        if args.url:
            summary = pipe.ingest_urls(args.sources, use_sync=args.sync)
        else:
            summary = pipe.ingest_paths(args.sources, reparse=args.reparse, use_sync=args.sync)
    finally:
        pipe.kb.close()
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


def cmd_query(args: argparse.Namespace) -> int:
    # Searching needs no Extend key, only the KB (and the embedder for hybrid).
    kb = KnowledgeBase(_kb_path(args.db))
    embedder = None if args.keyword_only else embedder_from_env()
    try:
        hits = retrieve(args.text, kb, embedder=embedder, limit=args.limit)
    finally:
        kb.close()
    for h in hits:
        how = h.get("match", "keyword")
        if h.get("vector_score") is not None:
            how += f", cos {h['vector_score']:.2f}"
        print(f"--- {h['doc_source']}  (chunk {h['ordinal']}, pages {h['pages_json']}, {how})")
        print(h["snippet"].replace("<b>", "").replace("</b>", ""))
        print()
    if not hits:
        print("No matches.")
    return 0


def _kb_path(db: str | None) -> str:
    return db or os.environ.get("TINKER_KB_PATH", "./kb.sqlite3")


def cmd_stats(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(_kb_path(args.db))
    try:
        print(json.dumps(kb.stats(), indent=2))
    finally:
        kb.close()
    return 0


def cmd_ingest_wave(args: argparse.Namespace) -> int:
    """Phase 1: ingest a named wave with parallel workers and telemetry."""
    cfg = load_config(kb_path=args.db)

    def client_factory() -> ExtendClient:
        return ExtendClient(
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            base_url=cfg.base_url,
            workspace_id=cfg.workspace_id,
        )

    paths = iter_candidate_files([args.dir])
    if args.limit:
        paths = paths[: args.limit]
    ingester = WaveIngester(client_factory, cfg.kb_path, workers=args.workers,
                            embedder=embedder_from_env())

    def on_progress(progress: dict, res: dict) -> None:
        p = progress
        print(f"[{p['done'] + p['failed'] + p['skipped']}/{p['total']}]"
              f" ok={p['done']} failed={p['failed']} skipped={p['skipped']}"
              f" :: {res['source']} -> {res['status']}", flush=True)

    summary = ingester.run(paths, args.wave, progress_cb=on_progress)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    for r in summary["results"]:
        if r["status"] == "failed":
            print(f"FAILED: {r['source']}: {r.get('error')}")
        elif r.get("flags"):
            print(f"FLAGS [{r['source']}]: {'; '.join(r['flags'])}")
    return 1 if summary["failed"] else 0


def cmd_eval(args: argparse.Namespace) -> int:
    from tinker_extend.eval import compare_modes, load_eval_set, run_eval

    kb = KnowledgeBase(_kb_path(args.db))
    embedder = None if args.keyword_only else embedder_from_env()
    try:
        cases = load_eval_set(args.eval_file)
        if args.compare:
            cmp = compare_modes(kb, cases, embedder_from_env(), k=args.k)
            print(f"Golden set: {cmp['cases']} cases, k={cmp['k']}, model={cmp['model']}")
            print(f"{'mode':<9} {'hit_rate':>8} {'mrr':>6} {'answers':>8} {'abstain':>8}")
            for mode in ("keyword", "hybrid"):
                m = cmp[mode]
                if m is None:
                    print(f"{mode:<9} (embeddings disabled)")
                    continue
                print(f"{mode:<9} {m['hit_rate']:>8.2f} {m['mrr']:>6.2f}"
                      f" {m['answer_mean']:>8.2f} {m['abstention_rate']:>8.0%}")
                for q in m["misses"]:
                    print(f"  MISS ({mode}): {q}")
            return 0
        report = run_eval(kb, cases, llm=None, k=args.k, embedder=embedder)
    finally:
        kb.close()
    r, a, p = report["retrieval"], report["answers"], report["parse_health"]
    print(f"Parse health: {p['kb_stats']['documents']} docs, "
          f"{p['kb_stats']['chunks']} chunks, ~{p['kb_stats']['token_estimate']} tokens")
    for d in p["documents"]:
        print(f"  - {d['source']}: {d['chunks']} chunks, {d['pages']} pages, "
              f"{d['empty_chunks']} empty, {d['chunks_with_tables']} with tables")
    print(f"Retrieval @{r['k']}: hit_rate={r['hit_rate']:.2f}  mrr={r['mrr']:.2f}")
    for q in r["misses"]:
        print(f"  MISS: {q}")
    print(f"Answers: mean_score={a['mean_score']:.2f}")
    for c in a["per_case"]:
        mark = "ok" if c["score"] >= 1.0 else "PARTIAL" if c["score"] > 0 else "FAIL"
        print(f"  [{mark}] {c['question'][:70]}")
    print(json.dumps(report, indent=2)[:2000])
    return 0


def cmd_embed_backfill(args: argparse.Namespace) -> int:
    """Embed stored chunks that have no vector yet. No Extend API calls."""
    embedder = embedder_from_env()
    if isinstance(embedder, NullEmbedder):
        print("Embeddings are disabled or the model could not be loaded.")
        return 1
    kb = KnowledgeBase(_kb_path(args.db))
    try:
        n = backfill_embeddings(kb, embedder)
        print(json.dumps({"embedded_now": n, **kb.embedding_stats(embedder.model)}, indent=2))
    finally:
        kb.close()
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Remove documents from the KB (chunks, search entries, telemetry)."""
    kb = KnowledgeBase(_kb_path(args.db))
    try:
        missing = 0
        for doc_id in args.doc_ids:
            doc = kb.get_document(doc_id)
            if not doc:
                print(f"not found: {doc_id}")
                missing += 1
                continue
            kb.delete_document(doc_id)
            print(f"deleted: {doc_id} ({doc['source']})")
    finally:
        kb.close()
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tinker-extend", description="Extend-powered knowledge base")
    ap.add_argument("--db", default=None, help="SQLite KB path (default: $TINKER_KB_PATH or ./kb.sqlite3)")
    sub = ap.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Parse documents into the knowledge base")
    p_ing.add_argument("sources", nargs="+", help="Files and/or directories to ingest")
    p_ing.add_argument("--url", action="store_true", help="Treat sources as URLs, not local paths")
    p_ing.add_argument("--reparse", action="store_true", help="Re-parse even if content hash is unchanged")
    p_ing.add_argument("--sync", action="store_true", help="Use the sync parse endpoint (small docs only)")
    p_ing.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="Full-text search the knowledge base")
    p_q.add_argument("text", help="Search query")
    p_q.add_argument("--limit", type=int, default=10)
    p_q.add_argument("--keyword-only", action="store_true", help="Skip vector search")
    p_q.set_defaults(func=cmd_query)

    p_s = sub.add_parser("stats", help="Show KB document/chunk counts")
    p_s.set_defaults(func=cmd_stats)

    p_e = sub.add_parser("eval", help="Run the evaluation framework against a golden set")
    p_e.add_argument("--eval-file", required=True, help="Path to golden Q&A JSONL")
    p_e.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics")
    p_e.add_argument("--keyword-only", action="store_true", help="Evaluate keyword search only")
    p_e.add_argument("--compare", action="store_true",
                     help="Report keyword-only vs hybrid side by side")
    p_e.set_defaults(func=cmd_eval)

    p_w = sub.add_parser("ingest-wave", help="Phase 1: ingest a named wave (parallel, telemetry, dead-letter)")
    p_w.add_argument("--dir", required=True, help="Directory of documents to ingest")
    p_w.add_argument("--wave", required=True, help="Wave name (e.g. pilot-100)")
    p_w.add_argument("--limit", type=int, default=0, help="Max files to take (0 = all)")
    p_w.add_argument("--workers", type=int, default=4, help="Parallel parse workers")
    p_w.set_defaults(func=cmd_ingest_wave)

    p_b = sub.add_parser("embed-backfill", help="Embed stored chunks that have no vector yet")
    p_b.set_defaults(func=cmd_embed_backfill)

    p_d = sub.add_parser("delete", help="Remove documents from the knowledge base by doc_id")
    p_d.add_argument("doc_ids", nargs="+", help="Document ids (see `stats` or the Fidelity page URL)")
    p_d.set_defaults(func=cmd_delete)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
