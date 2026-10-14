"""Web UI: Perplexity-style Q&A over the knowledge base.

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
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .admin import router as admin_router
from .classify import classify_content
from .config import load_config
from .extend_client import ExtendClient
from .fidelity import fidelity_metrics
from .htmlfmt import to_safe_html
from .ingest import iter_candidate_files
from .llm import llm_from_env
from .pages import page_boxes, render_pages, resolve_pdf
from .pipeline import KnowledgeBasePipeline
from .qa import ask
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
        pipe = KnowledgeBasePipeline(client, kb)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("tinkerExtend boot:", seed_if_empty(), flush=True)
    print("tinkerExtend boot:", ensure_page_images(), flush=True)
    print("tinkerExtend boot:", backfill_telemetry(), flush=True)
    yield


app = FastAPI(title="tinkerExtend Q&A", lifespan=lifespan)
APP_VERSION = "0.6.0"  # admin console: Phase 0/1 visibility
os.makedirs(PAGE_IMG_DIR, exist_ok=True)
app.mount("/page_images", StaticFiles(directory=PAGE_IMG_DIR), name="page_images")
app.include_router(admin_router)


@app.get("/api/stats")
def api_stats():
    kb = get_kb()
    try:
        return kb.stats()
    finally:
        kb.close()


@app.get("/api/search")
def api_search(q: str, limit: int = 10):
    kb = get_kb()
    try:
        hits = kb.search(q, limit=limit)
        passages = [
            {"n": i + 1, "chunk_id": h["chunk_id"], "doc_source": h["doc_source"],
             "pages": h["pages_json"], "content": h["content"]}
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
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    kb = get_kb()
    try:
        result = ask(question, kb, llm=llm_from_env(), limit=limit)
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
        )
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


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tinkerExtend Q&A</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
         background: #0e1116; color: #e6e9ef; margin: 0; padding: 20px 24px; }
  .wrap { max-width: 1400px; margin: 0 auto; }
  header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; }
  h1 { font-size: 22px; margin: 0; }
  .sub { color: #8b93a3; font-size: 13px; }
  .box { display: flex; gap: 8px; margin-bottom: 18px; max-width: 900px; }
  input[type=text] { flex: 1; padding: 12px 14px; font-size: 15px; border-radius: 10px;
    border: 1px solid #2a3140; background: #161b24; color: #e6e9ef; }
  button.go { padding: 12px 20px; font-size: 15px; border-radius: 10px; border: 0;
    background: #2f6fed; color: white; cursor: pointer; font-weight: 600; }
  button.ghost { background: #1c2330; border: 1px solid #2a3140; color: #e6e9ef;
    padding: 12px 20px; font-size: 15px; border-radius: 10px; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .status { color: #8b93a3; font-size: 13px; margin: 10px 0; min-height: 18px; }
  .layout { display: grid; grid-template-columns: minmax(360px, 5fr) 7fr; gap: 20px;
    align-items: start; }
  /* ---- source viewer (left) ---- */
  .viewer { position: sticky; top: 16px; background: #10141b;
    border: 1px solid #232b3b; border-radius: 12px; padding: 14px; }
  .viewer .vhead { font-size: 12px; color: #8b93a3; margin-bottom: 10px;
    display: flex; justify-content: space-between; gap: 8px; }
  .viewer .vhead b { color: #c8cfdb; font-weight: 600; }
  .pagewrap { position: relative; background: #fff; border-radius: 6px; overflow: hidden; }
  .pagewrap img { display: block; width: 100%; height: auto; }
  .hl { position: absolute; background: rgba(255, 196, 0, .28);
    border: 2px solid rgba(255, 170, 0, .85); border-radius: 2px;
    pointer-events: none; }
  .pager { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
  .pager button { background: #1c2330; border: 1px solid #2a3140; color: #e6e9ef;
    border-radius: 8px; padding: 6px 12px; cursor: pointer; font-size: 13px; }
  .pager .pg { font-size: 12px; color: #8b93a3; }
  .thumbs { display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap; }
  .thumbs button { background: #1c2330; border: 1px solid #2a3140; color: #9fb3d9;
    border-radius: 8px; padding: 5px 10px; cursor: pointer; font-size: 12px; }
  .thumbs button.active { border-color: #2f6fed; color: #fff; background: #22345c; }
  .vempty { color: #5c6577; font-size: 13px; text-align: center; padding: 40px 10px; }
  /* ---- answer column (right) ---- */
  .answer { background: #141a26; border: 1px solid #2a3a5f; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 14px; line-height: 1.6; font-size: 15px; }
  .answer .mode { font-size: 11px; color: #8b93a3; text-transform: uppercase;
    letter-spacing: .08em; margin-bottom: 10px; }
  .answer table { border-collapse: collapse; margin: 10px 0; font-size: 13px;
    display: block; overflow-x: auto; }
  .answer th, .answer td { border: 1px solid #3a4358; padding: 6px 10px; text-align: left; }
  .answer th { background: #1d2536; }
  .answer h2, .answer h3, .answer h4 { margin: 14px 0 6px; font-size: 16px; }
  .cite { background: #22345c; border: 1px solid #2f6fed; color: #bcd0ff;
    border-radius: 6px; padding: 1px 7px; font-size: 12px; cursor: pointer; margin: 0 1px; }
  .cite:hover { background: #2f6fed; color: #fff; }
  .feedback { margin-top: 14px; padding-top: 12px; border-top: 1px solid #232b3b; }
  .feedback .flabel { font-size: 12px; color: #8b93a3; margin-bottom: 8px; }
  .feedback .frow { display: flex; gap: 8px; align-items: center; }
  .fb { background: #1c2330; border: 1px solid #2a3140; color: #e6e9ef;
    border-radius: 8px; padding: 7px 14px; cursor: pointer; font-size: 14px; }
  .fb.sel-up { border-color: #2fa84f; background: #14301f; }
  .fb.sel-down { border-color: #d64545; background: #331414; }
  .feedback textarea { width: 100%; margin-top: 8px; background: #161b24; color: #e6e9ef;
    border: 1px solid #2a3140; border-radius: 8px; padding: 8px; font-size: 13px;
    font-family: inherit; resize: vertical; }
  .feedback .thanks { color: #7fd08a; font-size: 13px; }
  .passage { background: #131720; border: 1px solid #232b3b; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px; cursor: pointer; }
  .passage:hover { border-color: #2f6fed; }
  .passage.active { border-color: #2f6fed; box-shadow: 0 0 0 1px #2f6fed; }
  .passage .meta { font-size: 12px; color: #8b93a3; margin-bottom: 8px; }
  .passage .meta b { color: #c8cfdb; }
  .passage .pbody { font-size: 13.5px; line-height: 1.55; max-height: 200px;
    overflow: auto; color: #c9d1de; }
  .passage .pbody table { border-collapse: collapse; margin: 8px 0; font-size: 12px;
    display: block; overflow-x: auto; }
  .passage .pbody th, .passage .pbody td { border: 1px solid #3a4358; padding: 5px 8px; }
  .passage .pbody th { background: #1d2536; }
  .passage .pbody h2, .passage .pbody h3, .passage .pbody h4 { font-size: 14px; margin: 10px 0 4px; }
  @media (max-width: 920px) {
    .layout { grid-template-columns: 1fr; }
    .viewer { position: static; order: 2; }
    .qacol { order: 1; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header><h1>tinkerExtend Q&amp;A</h1><div class="sub" id="stats">loading…</div>
    <div style="flex:1"></div><a href="/admin" style="color:#7ea2ff;font-size:13px;text-decoration:none">Console →</a></header>
  <div class="box">
    <input type="text" id="q" placeholder="Ask about your documents…" autocomplete="off">
    <button class="go" id="askBtn">Ask</button>
    <button class="ghost" id="searchBtn">Search</button>
  </div>
  <div class="status" id="status"></div>
  <div class="layout">
    <section class="viewer" id="viewer">
      <div class="vempty">Ask a question — the source page behind the answer will appear here.</div>
    </section>
    <section class="qacol">
      <div id="answer"></div>
      <div id="results"></div>
    </section>
  </div>
</div>
<script>
const q = document.getElementById('q');
const statusEl = document.getElementById('status');
const answerEl = document.getElementById('answer');
const resultsEl = document.getElementById('results');
const viewerEl = document.getElementById('viewer');
let current = null;   // {question, mode, answer_text, chunk_ids, passages}
let viewState = null; // {passageIdx, pageIdx}

fetch('/api/stats').then(r => r.json()).then(s => {
  document.getElementById('stats').textContent =
    s.documents + ' documents · ' + s.chunks + ' chunks · ~' + s.token_estimate + ' tokens';
});

function esc(t){ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }

function allSourcePages(){
  const out = [];
  (current.passages || []).forEach((p, pi) => {
    (p.sources || []).forEach((s, si) => {
      if (s.image_url) out.push({passageIdx: pi, pageIdx: si, n: p.n, page: s.page,
        image_url: s.image_url, boxes: s.boxes, width: s.width, height: s.height,
        doc: p.doc_name});
    });
  });
  return out;
}

function renderViewer(){
  const pages = allSourcePages();
  if (!pages.length) {
    viewerEl.innerHTML = '<div class="vempty">No page images for these sources.</div>';
    return;
  }
  if (!viewState || viewState.passageIdx >= current.passages.length) {
    viewState = {passageIdx: 0, pageIdx: 0};
  }
  // find the flat index matching viewState
  let flat = pages.findIndex(x => x.passageIdx === viewState.passageIdx && x.pageIdx === viewState.pageIdx);
  if (flat < 0) flat = 0;
  const cur = pages[flat];
  const boxes = (cur.boxes || []).map(b => {
    const l = (b.left / cur.width * 100).toFixed(2), t = (b.top / cur.height * 100).toFixed(2);
    const w = ((b.right - b.left) / cur.width * 100).toFixed(2);
    const h = ((b.bottom - b.top) / cur.height * 100).toFixed(2);
    return `<div class="hl" style="left:${l}%;top:${t}%;width:${w}%;height:${h}%"></div>`;
  }).join('');
  const thumbs = pages.map((x, i) =>
    `<button data-i="${i}" class="${i === flat ? 'active' : ''}">[${x.n}] p.${x.page}</button>`).join('');
  viewerEl.innerHTML =
    `<div class="vhead"><span><b>Source</b> · ${esc(cur.doc)}</span><span>p.${cur.page}</span></div>
     <div class="pagewrap"><img src="${esc(cur.image_url)}" alt="source page">${boxes}</div>
     <div class="pager">
       <button id="pgPrev">← Prev</button>
       <button id="pgNext">Next →</button>
       <span class="pg">${flat + 1} / ${pages.length} cited pages</span>
     </div>
     <div class="thumbs">${thumbs}</div>`;
  const go = i => {
    const t = pages[(i + pages.length) % pages.length];
    viewState = {passageIdx: t.passageIdx, pageIdx: t.pageIdx};
    markActive(t.passageIdx); renderViewer();
  };
  document.getElementById('pgPrev').onclick = () => go(flat - 1);
  document.getElementById('pgNext').onclick = () => go(flat + 1);
  viewerEl.querySelectorAll('.thumbs button').forEach(b => {
    b.onclick = () => go(parseInt(b.dataset.i, 10));
  });
}

function markActive(pi){
  resultsEl.querySelectorAll('.passage').forEach((el, i) => {
    el.classList.toggle('active', i === pi);
  });
}

function showSourceForPassage(pi, si){
  viewState = {passageIdx: pi, pageIdx: si || 0};
  markActive(pi); renderViewer();
  viewerEl.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

function feedbackWidget(){
  return `<div class="feedback" id="fb">
    <div class="flabel">Was this answer right? Your feedback trains the eval set.</div>
    <div class="frow">
      <button class="fb" id="fbUp" title="Helpful">👍 Yes</button>
      <button class="fb" id="fbDown" title="Not helpful">👎 No</button>
    </div>
    <textarea id="fbComment" rows="2" placeholder="What was wrong? (optional)"></textarea>
  </div>`;
}

function wireFeedback(){
  const box = document.getElementById('fb');
  if (!box || !current) return;
  const send = async rating => {
    const comment = document.getElementById('fbComment').value;
    const r = await fetch('/api/feedback', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: current.question, mode: current.mode,
        answer: current.answer_text, chunk_ids: current.chunk_ids,
        rating, comment})});
    if (r.ok) box.innerHTML = '<div class="thanks">Thanks — recorded. This helps catch bad answers.</div>';
  };
  document.getElementById('fbUp').onclick = e => {
    e.target.classList.add('sel-up'); send(1);
  };
  document.getElementById('fbDown').onclick = e => {
    e.target.classList.add('sel-down'); send(-1);
  };
}

function renderPassages(){
  resultsEl.innerHTML = (current.passages || []).map((p, i) =>
    `<div class="passage" data-i="${i}">
       <div class="meta"><b>[${p.n}]</b> ${esc(p.doc_name)} · ` +
       ((p.sources || []).map(s => 'p.' + s.page).join(', ') || 'no page info') + `</div>
       <div class="pbody">${p.content_html}</div>
     </div>`).join('');
  resultsEl.querySelectorAll('.passage').forEach(el => {
    el.onclick = () => showSourceForPassage(parseInt(el.dataset.i, 10), 0);
  });
  answerEl.querySelectorAll('.cite').forEach(c => {
    c.onclick = () => {
      const n = parseInt(c.dataset.n, 10);
      const pi = (current.passages || []).findIndex(p => p.n === n);
      if (pi >= 0) showSourceForPassage(pi, 0);
    };
  });
}

async function doAsk(){
  const question = q.value.trim();
  if (!question) return;
  statusEl.textContent = 'Thinking…'; answerEl.innerHTML = ''; resultsEl.innerHTML = '';
  viewerEl.innerHTML = '<div class="vempty">Finding sources…</div>';
  const r = await fetch('/api/ask', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, limit: 5})});
  const data = await r.json();
  statusEl.textContent = '';
  if (data.error) { statusEl.textContent = data.error; return; }
  current = data; viewState = null;
  answerEl.innerHTML = `<div class="answer"><div class="mode">answer · ${esc(data.mode)}</div>` +
    data.answer_html +
    (data.note ? `<div class="mode" style="margin-top:10px">${esc(data.note)}</div>` : '') +
    feedbackWidget() + `</div>`;
  wireFeedback();
  renderPassages();
  renderViewer();
}

async function doSearch(){
  const query = q.value.trim();
  if (!query) return;
  statusEl.textContent = 'Searching…'; answerEl.innerHTML = ''; resultsEl.innerHTML = '';
  const r = await fetch('/api/search?q=' + encodeURIComponent(query) + '&limit=10');
  const data = await r.json();
  current = {question: query, mode: 'search', answer_text: '', chunk_ids: [],
    passages: data.hits.map((h, i) => ({...h, n: i + 1}))};
  viewState = null;
  statusEl.textContent = data.hits.length + ' matches — click a passage to see its source page';
  renderPassages();
  renderViewer();
}

document.getElementById('askBtn').onclick = doAsk;
document.getElementById('searchBtn').onclick = doSearch;
q.addEventListener('keydown', e => { if (e.key === 'Enter') doAsk(); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
