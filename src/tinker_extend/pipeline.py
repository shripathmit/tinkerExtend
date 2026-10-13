"""Orchestration: sources -> Extend parse -> chunks -> knowledge base."""
from __future__ import annotations

import json
import os
from dataclasses import replace

from .chunking import chunks_from_parse_run, num_pages_in_run
from .embeddings import Embedder, NullEmbedder
from .extend_client import ExtendAPIError, ExtendClient, ExtendRunFailed
from .ingest import iter_candidate_files, sha256_file
from .models import Chunk, Document
from .store import KnowledgeBase


class KnowledgeBasePipeline:
    def __init__(
        self,
        client: ExtendClient,
        kb: KnowledgeBase,
        embedder: Embedder | None = None,
        poll_interval_s: float = 5.0,
        poll_timeout_s: float = 1800.0,
    ):
        self.client = client
        self.kb = kb
        self.embedder = embedder or NullEmbedder()
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s

    # -- single-document steps -------------------------------------------
    def _parse_local_file(self, path: str, use_sync: bool) -> tuple[str, dict]:
        file_id = self.client.upload_file(path)
        file_ref = {"id": file_id}
        if use_sync:
            run = self.client.parse_sync(file_ref)
        else:
            run = self.client.parse_async(
                file_ref,
                poll_interval_s=self.poll_interval_s,
                timeout_s=self.poll_timeout_s,
                metadata={"source": os.path.basename(path)},
            )
        return file_id, run

    def _parse_url(self, url: str, use_sync: bool) -> tuple[None, dict]:
        file_ref = {"url": url}
        if use_sync:
            run = self.client.parse_sync(file_ref)
        else:
            run = self.client.parse_async(
                file_ref,
                poll_interval_s=self.poll_interval_s,
                timeout_s=self.poll_timeout_s,
                metadata={"source": url},
            )
        return None, run

    def _store_run(
        self, doc_id: str, source: str, file_hash: str,
        extend_file_id: str | None, run: dict,
    ) -> Document:
        chunks = chunks_from_parse_run(run, doc_id)
        vectors = self.embedder.embed([c.content for c in chunks])
        stored: list[Chunk] = []
        for chunk, vec in zip(chunks, vectors):
            if vec is not None:
                chunk = replace(chunk, embedding_json=json.dumps(vec))
            stored.append(chunk)
        doc = Document(
            doc_id=doc_id,
            source=source,
            file_hash=file_hash,
            extend_file_id=extend_file_id,
            parse_run_id=run.get("id"),
            num_pages=num_pages_in_run(run),
            num_chunks=len(stored),
            metadata_json=json.dumps({"usage": run.get("usage")}),
        )
        self.kb.upsert_document(doc)
        self.kb.replace_chunks(doc.doc_id, stored)
        return doc

    # -- batch entry points ----------------------------------------------
    def ingest_paths(
        self, sources: list[str], reparse: bool = False, use_sync: bool = False
    ) -> dict:
        """Ingest local files/directories. Returns a summary dict."""
        summary = {"ingested": 0, "skipped": 0, "failed": 0, "errors": []}
        for path in iter_candidate_files(sources):
            file_hash = sha256_file(path)
            if self.kb.has_hash(file_hash) and not reparse:
                summary["skipped"] += 1
                continue
            try:
                extend_file_id, run = self._parse_local_file(path, use_sync)
                self._store_run(file_hash, path, file_hash, extend_file_id, run)
                summary["ingested"] += 1
            except (ExtendAPIError, ExtendRunFailed, TimeoutError, OSError) as exc:
                summary["failed"] += 1
                summary["errors"].append({"source": path, "error": str(exc)})
        return summary

    def ingest_urls(self, urls: list[str], use_sync: bool = False) -> dict:
        """Ingest documents hosted at URLs (no upload step)."""
        summary = {"ingested": 0, "skipped": 0, "failed": 0, "errors": []}
        for url in urls:
            try:
                _file_id, run = self._parse_url(url, use_sync)
                doc_id = f"url:{run.get('id')}"
                self._store_run(doc_id, url, doc_id, None, run)
                summary["ingested"] += 1
            except (ExtendAPIError, ExtendRunFailed, TimeoutError) as exc:
                summary["failed"] += 1
                summary["errors"].append({"source": url, "error": str(exc)})
        return summary

    def query(self, text: str, limit: int = 10) -> list[dict]:
        return self.kb.search(text, limit=limit)
