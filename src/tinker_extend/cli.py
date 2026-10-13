"""Command line: python -m tinker_extend.cli {ingest|query|stats} ..."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinker_extend.config import load_config
from tinker_extend.extend_client import ExtendClient
from tinker_extend.pipeline import KnowledgeBasePipeline
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
    pipe = build_pipeline(args.db)
    try:
        hits = pipe.query(args.text, limit=args.limit)
    finally:
        pipe.kb.close()
    for h in hits:
        print(f"--- {h['doc_source']}  (chunk {h['ordinal']}, pages {h['pages_json']})")
        print(h["snippet"].replace("<b>", "").replace("</b>", ""))
        print()
    if not hits:
        print("No matches.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(args.db or load_config().kb_path)
    try:
        print(json.dumps(kb.stats(), indent=2))
    finally:
        kb.close()
    return 0


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
    p_q.set_defaults(func=cmd_query)

    p_s = sub.add_parser("stats", help="Show KB document/chunk counts")
    p_s.set_defaults(func=cmd_stats)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
