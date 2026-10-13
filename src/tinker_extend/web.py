"""Web UI: search and Q&A over the knowledge base.

Run locally:  uvicorn tinker_extend.web:app --reload
Railway:       Procfile + railway.json are included.
"""
from __future__ import annotations

import html
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .config import load_config
from .extend_client import ExtendClient
from .ingest import iter_candidate_files
from .llm import llm_from_env
from .pipeline import KnowledgeBasePipeline
from .qa import ask
from .store import KnowledgeBase

SEED_DIR = os.environ.get("TINKER_SEED_DIR", "./seed_docs")


def get_kb() -> KnowledgeBase:
    # Serving search/Q&A needs no API key — only ingest does.
    return KnowledgeBase(os.environ.get("TINKER_KB_PATH", "./kb.sqlite3"))


def seed_if_empty() -> str:
    """Ingest seed_docs on first boot (Railway) when a key is available."""
    kb_path = os.environ.get("TINKER_KB_PATH", "./kb.sqlite3")
    kb = KnowledgeBase(kb_path)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("tinkerExtend boot:", seed_if_empty(), flush=True)
    yield


app = FastAPI(title="tinkerExtend Q&A", lifespan=lifespan)


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
        return {"query": q, "hits": kb.search(q, limit=limit)}
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
        return ask(question, kb, llm=llm_from_env(), limit=limit)
    finally:
        kb.close()


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tinkerExtend Q&A</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #0e1116;
         color: #e6e9ef; margin: 0; padding: 24px; }
  .wrap { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: #8b93a3; font-size: 13px; margin-bottom: 20px; }
  .box { display: flex; gap: 8px; margin-bottom: 16px; }
  input[type=text] { flex: 1; padding: 12px; font-size: 15px; border-radius: 8px;
    border: 1px solid #2a3140; background: #161b24; color: #e6e9ef; }
  button { padding: 12px 18px; font-size: 15px; border-radius: 8px; border: 0;
    background: #2f6fed; color: white; cursor: pointer; }
  button.ghost { background: #1c2330; border: 1px solid #2a3140; }
  button:disabled { opacity: .5; cursor: default; }
  .answer { background: #141a26; border: 1px solid #2a3a5f; border-radius: 10px;
    padding: 16px; margin-bottom: 12px; white-space: pre-wrap; line-height: 1.55; }
  .answer .mode { font-size: 11px; color: #8b93a3; text-transform: uppercase;
    letter-spacing: .08em; margin-bottom: 8px; }
  .passage { background: #131720; border: 1px solid #232b3b; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px; }
  .passage .meta { font-size: 12px; color: #8b93a3; margin-bottom: 6px; }
  .passage .body { font-size: 14px; line-height: 1.55; white-space: pre-wrap;
    max-height: 220px; overflow: auto; }
  .status { color: #8b93a3; font-size: 13px; margin: 12px 0; }
  details { margin-top: 8px; }
  summary { cursor: pointer; color: #8b93a3; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>tinkerExtend Q&A</h1>
  <div class="sub" id="stats">loading knowledge base…</div>
  <div class="box">
    <input type="text" id="q" placeholder="Ask about your documents…" autocomplete="off">
    <button id="askBtn">Ask</button>
    <button id="searchBtn" class="ghost">Search</button>
  </div>
  <div class="status" id="status"></div>
  <div id="answer"></div>
  <div id="results"></div>
</div>
<script>
const q = document.getElementById('q');
const statusEl = document.getElementById('status');
const answerEl = document.getElementById('answer');
const resultsEl = document.getElementById('results');

fetch('/api/stats').then(r => r.json()).then(s => {
  document.getElementById('stats').textContent =
    s.documents + ' documents · ' + s.chunks + ' chunks · ~' + s.token_estimate + ' tokens';
});

function esc(t){ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }

function renderPassages(passages){
  resultsEl.innerHTML = passages.map(p =>
    '<div class="passage"><div class="meta">[' + p.n + '] ' + esc(p.doc_source.split('/').pop()) +
    ' · pages ' + esc(p.pages) + '</div><div class="body">' + esc(p.content) + '</div></div>'
  ).join('');
}

async function doAsk(){
  const question = q.value.trim();
  if(!question) return;
  statusEl.textContent = 'Thinking…'; answerEl.innerHTML=''; resultsEl.innerHTML='';
  const r = await fetch('/api/ask', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({question, limit: 5})});
  const data = await r.json();
  statusEl.textContent = '';
  if(data.error){ statusEl.textContent = data.error; return; }
  answerEl.innerHTML = '<div class="answer"><div class="mode">answer · ' +
    esc(data.mode) + '</div>' + esc(data.answer) +
    (data.note ? '<div class="mode" style="margin-top:8px">' + esc(data.note) + '</div>' : '') + '</div>';
  renderPassages(data.passages || []);
}

async function doSearch(){
  const query = q.value.trim();
  if(!query) return;
  statusEl.textContent = 'Searching…'; answerEl.innerHTML=''; resultsEl.innerHTML='';
  const r = await fetch('/api/search?q=' + encodeURIComponent(query) + '&limit=10');
  const data = await r.json();
  statusEl.textContent = data.hits.length + ' matches';
  renderPassages(data.hits.map((h,i) => ({n:i+1, doc_source:h.doc_source,
    pages:h.pages_json, content:h.content})));
}

document.getElementById('askBtn').onclick = doAsk;
document.getElementById('searchBtn').onclick = doSearch;
q.addEventListener('keydown', e => { if(e.key === 'Enter') doAsk(); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
