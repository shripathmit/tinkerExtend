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

## Tests

```bash
python -m pytest tests/ -q
```
