# tinkerExtend

A document-ingestion pipeline that turns a folder of documents into a searchable
knowledge base, powered by the [Extend](https://www.extend.ai) document-processing API.

Pipeline: **collect → upload → parse (Extend) → normalize chunks → embed → SQLite knowledge base (FTS + vectors) → hybrid query**

## Setup

```bash
pip install -r requirements.txt
export EXTEND_API_KEY="your-extend-api-key"   # create one at dashboard.extend.ai → Developers
```

Optional:

```bash
export EXTEND_API_VERSION="2026-02-09"   # default; pin to avoid surprise upgrades
export EXTEND_WORKSPACE_ID="..."        # only needed for organization-scoped keys
export TINKER_KB_PATH="./kb.sqlite3"    # default knowledge-base location
```

## Usage

Ingest a folder of documents (PDFs, Word, Excel, images, …):

```bash
python -m tinker_extend.cli ingest ./docs --db ./kb.sqlite3
```

Re-ingest skips files whose content hash is unchanged. Force with `--reparse`.
For large files the pipeline uses async parse runs with polling; `--sync` forces
the synchronous endpoint (5-minute timeout, best for small docs).

Ask the knowledge base:

```bash
python -m tinker_extend.cli query "What is the refund policy?" --db ./kb.sqlite3
python -m tinker_extend.cli stats --db ./kb.sqlite3
```

## How it works

- `extend_client.py` — thin wrapper over the Extend REST API (`urllib`, no SDK
  dependency): file upload, sync parse, and async parse runs with polling.
- `chunking.py` — normalizes Extend's page chunks (markdown + typed layout blocks)
  into KB records, with optional token-window re-chunking.
- `store.py` — SQLite knowledge base: `documents` and `chunks` tables plus an
  FTS5 full-text index, so `query` works with zero extra infrastructure.
- `embeddings.py` — local embedding model (fastembed, `BAAI/bge-small-en-v1.5`)
  behind a pluggable interface; falls back to keyword-only search if disabled.
- `pipeline.py` — orchestration with content-hash delta updates.

## Hybrid search — keywords + embeddings

Every chunk is embedded with a small local model (no API key, no per-call
cost) and stored as a vector next to the text. A query runs both full-text
search (exact figures, rare terms) and vector search (paraphrases, synonyms),
then merges the two rankings with reciprocal rank fusion.

- The model downloads from HuggingFace on first use (~67MB). Point
  `TINKER_EMBED_CACHE` at a persistent volume so redeploys reuse it.
- Documents ingested before embeddings existed are backfilled automatically
  when the web app starts, or on demand: `python -m tinker_extend.cli embed-backfill`.
- Changing `TINKER_EMBED_MODEL` makes old vectors stale; they are ignored
  until the backfill re-embeds them.
- A question is answered only if the top passage shares key terms with it
  or its vector similarity is at least `TINKER_MIN_SIM` (default 0.80);
  otherwise the system says it couldn't find an answer.
- `TINKER_EMBEDDINGS=off` turns it all off (keyword search only).

Compare the two modes on the golden set (also shown in `/admin` → Golden Sets):

```bash
python -m tinker_extend.cli --db ./kb.sqlite3 eval --eval-file eval/golden_10q.jsonl --compare
python -m tinker_extend.cli query "How much did R&D spending fall?" --keyword-only
```

## Web app — one UI, five tabs

```bash
uvicorn --app-dir src tinker_extend.web:app --host 0.0.0.0 --port 8000
```

| Tab | Access | What it does |
|---|---|---|
| **Q&A** | public | Ask (grounded answer with citations) or Search. The viewer shows the real page with the cited region boxed; *Surface* switches to a text reader (dark or paper), *All chunks* outlines every chunk on the page. *Accurate? Yes/No* turns the answer into a golden assertion. |
| **Overview** | code | Documents, parse success, golden pass rate, spend; red flags with *Inspect & Fix*. |
| **Ingestion** | code | *+ Ingest Document* uploads files and runs a wave with live progress; waves by scale tier; corpus inventory; dead letters (retry/resolve). |
| **Parse Fidelity** | code | Table shape, page-number leaks, text coverage per document. *Inspect* shows the problem passages and re-parses with Extend's Performance engine (spends credits, ~$0.025/page), then shows before → after. |
| **Golden Sets** | code | Every assertion with live PASS/FAIL: the golden Q&A file, golden-doc probes, and Q&A feedback (thumbs-up: the same chunk must stay in the top 5; thumbs-down: FAIL until resolved). |

The header's **LLM** switch picks extractive or generated answers. Generated
answers need `TINKER_LLM_API_KEY` (plus optional `TINKER_LLM_BASE_URL` /
`TINKER_LLM_MODEL`, any OpenAI-compatible endpoint); without it answers are
extractive — the best passage, quoted.

The console tabs ask for the access code once (default `extend`; set
`TINKER_ADMIN_CODE` to change it). Every `/api/admin/*` endpoint needs the
session cookie or an `X-Admin-Code` header:

```bash
curl -H "X-Admin-Code: $TINKER_ADMIN_CODE" https://<host>/api/admin/overview
```

The UI is static (`src/tinker_extend/static/`: `index.html`, `app.js`, and a
compiled Tailwind `app.css`), so the server needs no Node. After editing the
markup or classes, rebuild the CSS:

```bash
npx tailwindcss@3.4.17 -c src/tinker_extend/static/tailwind.config.js \
  -i src/tinker_extend/static/src/input.css -o src/tinker_extend/static/app.css --minify
```

To remove a document, use **Delete** on its row in the Fidelity matrix
(asks to confirm). This removes its passages, search entries, fidelity
scores, re-parse history and page images, plus the source PDF if it was an
upload (seed files are kept). Scripts can call
`DELETE /api/admin/documents/{doc_id}` or
`python -m tinker_extend.cli delete <doc_id>`.

Re-parse needs the original file on the server (its stored path, the seed
dir, or `TINKER_UPLOAD_DIR`), verified by content hash.

Deploy: `Procfile` + `railway.json` are included. On first boot the app seeds
itself from `./seed_docs` if the KB is empty and `EXTEND_API_KEY` is set.

## Evaluation — how good is the system?

Three layers, all runnable offline against your KB:

1. **Parse health** — per-document metrics (chunks, pages, empty chunks,
   tables detected) straight from ingest metadata.
2. **Retrieval quality** — hit_rate@k and MRR over a golden Q&A set.
3. **Answer quality** — expected facts present in answers; unanswerable
   questions must abstain, not hallucinate.

```bash
python -m tinker_extend.cli --db ./kb.sqlite3 eval --eval-file eval/golden_10q.jsonl
```

Add cases to `eval/` as you ingest documents. Re-run on every change —
a drop in hit_rate or abstention is a regression.

## Tests

```bash
python -m pytest tests/ -q
TINKER_TEST_REAL_EMBED=1 python -m pytest tests/test_embeddings.py -q   # also downloads + tests the real model
```
