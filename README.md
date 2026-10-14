# tinkerExtend

A document-ingestion pipeline that turns a folder of documents into a searchable
knowledge base, powered by the [Extend](https://www.extend.ai) document-processing API.

Pipeline: **collect → upload → parse (Extend) → normalize chunks → SQLite FTS knowledge base → query**

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
- `embeddings.py` — pluggable embedder interface. Ships with a null embedder;
  add your provider to enable vector search later.
- `pipeline.py` — orchestration with content-hash delta updates.

## Web UI — search and Q&A

```bash
uvicorn --app-dir src tinker_extend.web:app --host 0.0.0.0 --port 8000
```

Open the page, type a question, and hit **Ask** (grounded Q&A with citations)
or **Search** (raw full-text matches). Answers are extractive by default —
the top passages, no hallucination. Set `TINKER_LLM_API_KEY` (plus optional
`TINKER_LLM_BASE_URL` / `TINKER_LLM_MODEL`) to get generated answers from any
OpenAI-compatible chat endpoint, still grounded and cited.

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
```
