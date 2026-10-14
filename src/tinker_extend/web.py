"""Web app: the unified tinkerExtend UI (static/) and its public JSON API.

Left pane: the source document page the answer came from, with the exact
regions highlighted. Right pane: the answer with citations, plus a
feedback control so users can flag wrong answers. Feedback is stored and
exportable for growing the golden eval set.

Run locally:  uvicorn tinker_extend.web:app --reload
Railway:       Procfile + railway.json are included.
"""
from __future__ import annotations

import html
import json
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .admin import router as admin_router
from .classify import classify_content
from .config import load_config
from .embeddings import embedder_from_env
from .extend_client import ExtendClient
from .fidelity import fidelity_metrics
from .htmlfmt import to_safe_html
from .ingest import iter_candidate_files
from .llm import llm_from_env
from .pages import page_boxes, render_pages, resolve_pdf
from .pipeline import KnowledgeBasePipeline, backfill_embeddings
from .qa import ask, retrieve
from .store import KnowledgeBase

SEED_DIR = os.environ.get("TINKER_SEED_DIR", "./seed_docs")
KB_PATH = os.environ.get("TINKER_KB_PATH", "./kb.sqlite3")
PAGE_IMG_DIR = os.environ.get("TINKER_PAGE_IMG_DIR", "./page_images")


def get_kb() -> KnowledgeBase:
    # Serving search/Q&A needs no API key — only ingest does.
    return KnowledgeBase(KB_PATH)


def seed_if_empty() -> str:
    """Ingest seed_docs on first boot (Railway) when a key is available."""
    kb = KnowledgeBase(KB_PATH)
    try:
        if kb.stats()["documents"] > 0:
            return "kb already populated"
        if not os.path.isdir(SEED_DIR):
            return "no seed dir"
        files = iter_candidate_files([SEED_DIR])
        if not files:
            return "seed dir empty"
        try:
            cfg = load_config()
        except RuntimeError as exc:  # e.g. EXTEND_API_KEY not set
            return f"seed skipped: {exc}"
        client = ExtendClient(api_key=cfg.api_key, api_version=cfg.api_version,
                              base_url=cfg.base_url, workspace_id=cfg.workspace_id)
        pipe = KnowledgeBasePipeline(client, kb, embedder=embedder_from_env())
        summary = pipe.ingest_paths(files)
        return f"seeded: {summary}"
    finally:
        kb.close()


def ensure_page_images() -> str:
    """Render PDF pages to PNG once per document so the UI can show sources."""
    kb = get_kb()
    try:
        docs = [dict(r) for r in kb.conn.execute("SELECT doc_id, source FROM documents")]
    finally:
        kb.close()
    rendered = 0
    for doc in docs:
        out = os.path.join(PAGE_IMG_DIR, doc["doc_id"])
        if os.path.isdir(out) and os.listdir(out):
            continue
        pdf = resolve_pdf(doc["source"], SEED_DIR)
        if not pdf:
            continue
        try:
            render_pages(pdf, out)
            rendered += 1
        except Exception as exc:  # never break boot on rendering
            print(f"page render failed for {doc['doc_id']}: {exc}", flush=True)
    return f"page images ready for {rendered} new doc(s)"


def enrich_passage(kb: KnowledgeBase, p: dict) -> dict:
    """Attach source pages (image URLs + highlight boxes) to a passage."""
    row = kb.conn.execute(
        "SELECT blocks_json, doc_id FROM chunks WHERE chunk_id = ?",
        (p["chunk_id"],),
    ).fetchone()
    sources = []
    if row:
        for page_num in sorted(page_boxes(row["blocks_json"])):
            info = page_boxes(row["blocks_json"])[page_num]
            img = f"/page_images/{row['doc_id']}/p{page_num}.png"
            img_path = os.path.join(PAGE_IMG_DIR, row["doc_id"], f"p{page_num}.png")
            sources.append({
                "page": page_num,
                "image_url": img if os.path.exists(img_path) else None,
                "width": info["width"], "height": info["height"],
                "boxes": info["boxes"],
            })
    return {**p, "sources": sources,
            "doc_id": row["doc_id"] if row else None,
            "doc_name": os.path.basename(p["doc_source"]),
            "content_html": to_safe_html(p["content"])}


_CITE_RE = re.compile(r"\[(\d+)\]")


def linkify_citations(text: str) -> str:
    """Escape answer text and turn [n] markers into clickable citations."""
    parts = _CITE_RE.split(html.escape(text))
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # the captured number
            n = part
            out.append(f'<button class="cite" data-n="{n}" title="Show source">[{n}]</button>')
        else:
            out.append(part.replace("\n", "<br>"))
    return "".join(out)


def backfill_telemetry() -> str:
    """Compute fidelity metrics + telemetry for docs ingested before Phase 1.

    No API calls — everything comes from the stored blocks. Runs once;
    later docs get telemetry at ingest time.
    """
    kb = get_kb()
    try:
        existing = {r[0] for r in kb.conn.execute("SELECT doc_id FROM parse_telemetry")}
        n = 0
        for doc in kb.conn.execute("SELECT doc_id, source FROM documents"):
            if doc["doc_id"] in existing:
                continue
            rows = kb.conn.execute(
                "SELECT content, blocks_json FROM chunks WHERE doc_id = ?",
                (doc["doc_id"],)).fetchall()
            blocks: list = []
            for r in rows:
                try:
                    blocks.extend(json.loads(r["blocks_json"] or "[]"))
                except (json.JSONDecodeError, TypeError):
                    pass
            chunks = [r["content"] for r in rows]
            metrics = fidelity_metrics(blocks, chunks)
            doc_type, _, _ = classify_content("\n".join(chunks),
                                             os.path.basename(doc["source"]))
            kb.record_telemetry(doc["doc_id"], "backfill", doc_type, "default",
                                metrics, cost_usd=0.0, duration_s=0.0, status="ok")
            n += 1
        return f"backfilled telemetry for {n} doc(s)"
    finally:
        kb.close()


def recompute_fidelity_if_changed() -> str:
    """Re-run fidelity checks on stored blocks when their definition changed."""
    from .admin import recompute_doc_telemetry
    from .fidelity import FIDELITY_VERSION
    kb = get_kb()
    try:
        if kb.get_meta("fidelity_version") == FIDELITY_VERSION:
            return f"fidelity metrics current (v{FIDELITY_VERSION})"
        ids = [r[0] for r in kb.conn.execute("SELECT doc_id FROM parse_telemetry")]
        for doc_id in ids:
            recompute_doc_telemetry(kb, doc_id)
        kb.set_meta("fidelity_version", FIDELITY_VERSION)
        return f"recomputed fidelity for {len(ids)} doc(s) (v{FIDELITY_VERSION})"
    finally:
        kb.close()


def backfill_vectors() -> str:
    """Embed chunks stored before embeddings existed (or under another model)."""
    embedder = embedder_from_env()
    if not embedder.model:
        return "embeddings disabled; search is keyword-only"
    kb = get_kb()
    try:
        n = backfill_embeddings(kb, embedder)
        st = kb.embedding_stats(embedder.model)
        return (f"embedded {n} chunk(s); {st['embedded']}/{st['chunks']} have"
                f" {embedder.model} vectors")
    finally:
        kb.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model first so seeding stores vectors directly.
    embedder_from_env()
    print("tinkerExtend boot:", seed_if_empty(), flush=True)
    print("tinkerExtend boot:", ensure_page_images(), flush=True)
    print("tinkerExtend boot:", backfill_telemetry(), flush=True)
    print("tinkerExtend boot:", recompute_fidelity_if_changed(), flush=True)
    print("tinkerExtend boot:", backfill_vectors(), flush=True)
    yield


app = FastAPI(title="tinkerExtend Q&A", lifespan=lifespan)
APP_VERSION = __version__
os.makedirs(PAGE_IMG_DIR, exist_ok=True)
app.mount("/page_images", StaticFiles(directory=PAGE_IMG_DIR), name="page_images")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(admin_router)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"),
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/stats")
def api_stats():
    kb = get_kb()
    try:
        return {**kb.stats(), "embed_model": embedder_from_env().model,
                "llm_configured": llm_from_env() is not None, "version": __version__}
    finally:
        kb.close()


@app.get("/api/search")
def api_search(q: str, limit: int = 10):
    kb = get_kb()
    try:
        hits = retrieve(q, kb, embedder=embedder_from_env(), limit=limit)
        passages = [
            {"n": i + 1, "chunk_id": h["chunk_id"], "doc_source": h["doc_source"],
             "pages": h["pages_json"], "content": h["content"],
             "match": h.get("match", "keyword"), "vector_score": h.get("vector_score")}
            for i, h in enumerate(hits)
        ]
        return {"query": q,
                "hits": [enrich_passage(kb, p) for p in passages]}
    finally:
        kb.close()


@app.post("/api/ask")
def api_ask(payload: dict):
    question = (payload.get("question") or "").strip()
    limit = int(payload.get("limit") or 5)
    # synthesize=false forces an extractive answer even with an LLM configured.
    synthesize = payload.get("synthesize", True) is not False
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    kb = get_kb()
    try:
        embedder = embedder_from_env()
        result = ask(question, kb, llm=llm_from_env() if synthesize else None,
                     limit=limit, embedder=embedder)
        passages = [enrich_passage(kb, p) for p in result.get("passages", [])]
        if result["mode"] == "extractive":
            answer_html = to_safe_html(result["answer"])
        elif result["mode"] == "generated":
            answer_html = linkify_citations(result["answer"])
        else:
            answer_html = html.escape(result["answer"]).replace("\n", "<br>")
        return {
            "question": question,
            "mode": result["mode"],
            "answer_html": answer_html,
            "answer_text": result["answer"],
            "note": result.get("note", ""),
            "search": "hybrid" if embedder.model else "keyword",
            "passages": passages,
            "chunk_ids": [p["chunk_id"] for p in passages],
        }
    finally:
        kb.close()


@app.post("/api/feedback")
def api_feedback(payload: dict):
    rating = 1 if int(payload.get("rating") or 0) > 0 else -1
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    kb = get_kb()
    try:
        fid = kb.add_feedback(
            question=question,
            answer_mode=str(payload.get("mode") or ""),
            answer=str(payload.get("answer") or ""),
            chunk_ids=list(payload.get("chunk_ids") or []),
            rating=rating,
            comment=str(payload.get("comment") or ""),
            issue=str(payload.get("issue") or ""),
            active_chunk_id=str(payload.get("active_chunk_id") or ""),
        )
        kb.set_meta("golden_last_run", "")  # new assertion: next view re-runs
        return {"ok": True, "id": fid}
    finally:
        kb.close()


@app.get("/api/feedback/export")
def api_feedback_export():
    """Export feedback as JSONL — raw material for growing the golden set."""
    kb = get_kb()
    try:
        rows = kb.list_feedback()
    finally:
        kb.close()
    lines = [json.dumps({
        "question": r["question"], "answer_mode": r["answer_mode"],
        "rating": r["rating"], "comment": r["comment"],
        "chunk_ids": json.loads(r["chunk_ids"] or "[]"),
        "created_at": r["created_at"],
    }) for r in rows]
    return PlainTextResponse("\n".join(lines), media_type="application/x-ndjson")


@app.get("/api/examples")
def api_examples(n: int = 3):
    """A few answerable golden questions for the Q&A tab's "Try:" row."""
    from .admin import GOLDEN_QA
    from .eval import load_eval_set
    if not os.path.exists(GOLDEN_QA):
        return {"examples": []}
    cases = [c.question for c in load_eval_set(GOLDEN_QA) if c.answerable]
    return {"examples": cases[:max(0, min(n, 10))]}


@app.get("/api/page/{doc_id}/{page}")
def api_page(doc_id: str, page: int):
    """One source page: image, size, page count, and every chunk on it
    (with highlight boxes and safe HTML) for the viewer's image and
    reader modes."""
    kb = get_kb()
    try:
        doc = kb.get_document(doc_id)
        if not doc:
            return JSONResponse({"error": "doc not found"}, status_code=404)
        chunks, size = [], {"width": None, "height": None}
        for r in kb.conn.execute(
                "SELECT chunk_id, ordinal, content, blocks_json, pages_json FROM chunks"
                " WHERE doc_id = ? ORDER BY ordinal", (doc_id,)):
            try:
                pages = json.loads(r["pages_json"] or "[]")
            except (json.JSONDecodeError, TypeError):
                pages = []
            if page not in pages:
                continue
            info = page_boxes(r["blocks_json"]).get(page) or {}
            if info.get("width") and not size["width"]:
                size = {"width": info["width"], "height": info["height"]}
            chunks.append({"chunk_id": r["chunk_id"], "ordinal": r["ordinal"],
                           "boxes": info.get("boxes", []),
                           "content_html": to_safe_html(r["content"])})
        img_path = os.path.join(PAGE_IMG_DIR, doc_id, f"p{page}.png")
        num_pages = doc.get("num_pages") or 0
        if os.path.isdir(os.path.join(PAGE_IMG_DIR, doc_id)):
            num_pages = max(num_pages, sum(
                1 for f in os.listdir(os.path.join(PAGE_IMG_DIR, doc_id))
                if f.startswith("p") and f.endswith(".png")))
        return {"doc_id": doc_id, "doc_name": os.path.basename(doc["source"]),
                "page": page, "num_pages": num_pages,
                "image_url": f"/page_images/{doc_id}/p{page}.png" if os.path.exists(img_path) else None,
                **size, "chunks": chunks}
    finally:
        kb.close()
