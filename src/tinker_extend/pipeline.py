"""Orchestration: sources -> Extend parse -> chunks -> knowledge base."""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from .chunking import chunks_from_parse_run, num_pages_in_run
from .classify import classify_content, classify_filename, mark_scan
from .embeddings import Embedder, NullEmbedder, to_blob
from .extend_client import ExtendAPIError, ExtendClient, ExtendRunFailed
from .fidelity import fidelity_metrics, red_flags, text_coverage
from .ingest import iter_candidate_files, sha256_file
from .models import Chunk, Document
from .qa import retrieve
from .store import KnowledgeBase

# USD per page, Extend pay-as-you-go list prices (verified Oct 2026).
ENGINE_RATES = {"parse_light": 0.00625, "parse_performance": 0.025}

# Doc type -> parse engine. Tables/scans/handwriting earn Performance;
# text-heavy docs stay on Light. Tune after the 100-doc pilot wave.
DEFAULT_PROFILES = {
    "financial_filing": "parse_performance",
    "contract": "parse_performance",
    "invoice": "parse_performance",
    "scan": "parse_performance",
    "report": "parse_light",
    "email": "parse_light",
    "presentation": "parse_light",
    "other": "parse_light",
}


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
                chunk = replace(chunk, embedding=to_blob(vec),
                                embedding_model=self.embedder.model)
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
        return retrieve(text, self.kb, embedder=self.embedder, limit=limit)


def backfill_embeddings(kb: KnowledgeBase, embedder: Embedder,
                        batch_size: int = 64) -> int:
    """Embed chunks with no vector (or one from another model). No API calls.

    Safe to re-run: only missing/stale vectors are computed. Returns the
    number of chunks embedded.
    """
    if not embedder.model:
        return 0
    done = 0
    while True:
        rows = kb.chunks_needing_embedding(embedder.model, limit=batch_size)
        if not rows:
            return done
        vecs = embedder.embed([r["content"] for r in rows])
        updates = [(r["chunk_id"], to_blob(v), embedder.model)
                   for r, v in zip(rows, vecs) if v is not None]
        kb.set_embeddings(updates)
        done += len(updates)
        if len(updates) < len(rows):
            # Some texts got no vector; they would come back every batch.
            return done


def _source_text_words(path: str) -> str | None:
    """Best-effort source text via pypdf (optional dep); None if unavailable."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return None


class WaveIngester:
    """Phase 1: ingest a wave of documents with parallelism and resilience.

    Each document: classify (filename) -> pick engine -> upload+parse with
    retry/backoff -> chunk -> store -> fidelity metrics -> telemetry.
    Exhausted failures go to the dead-letter queue, never silently dropped.

    Workers are threads; each owns its own ExtendClient and KnowledgeBase
    connection because neither is thread-safe to share.
    """

    def __init__(
        self,
        client_factory,
        kb_path: str,
        workers: int = 4,
        profiles: dict | None = None,
        max_attempts: int = 3,
        backoff_s: float = 2.0,
        embedder: Embedder | None = None,
        force: bool = False,
        engine_override: str | None = None,
    ):
        """force: re-parse files already in the KB (same doc_id, chunks and
        telemetry replaced, before/after kept in reparse_history).
        engine_override: use this parse engine instead of the doc-type profile."""
        self.client_factory = client_factory
        self.kb_path = kb_path
        self.workers = workers
        self.profiles = profiles or DEFAULT_PROFILES
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.embedder = embedder or NullEmbedder()
        self.force = force
        self.engine_override = engine_override
        self._lock = threading.Lock()
        self.progress = {"done": 0, "failed": 0, "skipped": 0, "total": 0}

    def _bump(self, key: str) -> None:
        with self._lock:
            self.progress[key] += 1

    def _parse_with_retry(self, client: ExtendClient, path: str,
                          engine: str) -> tuple[str, dict]:
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                file_id = client.upload_file(path)
                run = client.parse_async(
                    {"id": file_id}, engine=engine,
                    metadata={"source": os.path.basename(path)})
                return file_id, run
            except ExtendAPIError as exc:
                last = exc
                if not exc.retryable or attempt == self.max_attempts:
                    break
            except (ExtendRunFailed, TimeoutError, OSError) as exc:
                last = exc
                if attempt == self.max_attempts:
                    break
            time.sleep(self.backoff_s * (2 ** (attempt - 1)))
        raise last  # type: ignore[misc]

    def _ingest_one(self, path: str, wave: str) -> dict:
        kb = KnowledgeBase(self.kb_path)
        client = self.client_factory()
        pipe = KnowledgeBasePipeline(client, kb, embedder=self.embedder)
        started = time.monotonic()
        try:
            file_hash = sha256_file(path)
            before = None
            if kb.has_hash(file_hash):
                if not self.force:
                    self._bump("skipped")
                    return {"source": path, "status": "skipped"}
                before = kb.telemetry_row(file_hash)
            pre_type, _, _ = classify_filename(os.path.basename(path))
            engine = self.engine_override or self.profiles.get(pre_type, "parse_light")
            extend_file_id, run = self._parse_with_retry(client, path, engine)
            doc = pipe._store_run(file_hash, path, file_hash, extend_file_id, run)
            # Blocks live per-chunk (run["output"]["chunks"][i]["blocks"]),
            # so collect them from the stored chunks — same as boot backfill.
            blocks: list = []
            chunks: list = []
            for r in kb.conn.execute(
                    "SELECT content, blocks_json FROM chunks WHERE doc_id=?",
                    (doc.doc_id,)).fetchall():
                chunks.append(r["content"])
                try:
                    blocks.extend(json.loads(r["blocks_json"] or "[]"))
                except (json.JSONDecodeError, TypeError):
                    pass
            metrics = fidelity_metrics(blocks, chunks)
            src_text = _source_text_words(path)
            if src_text:
                metrics["text_coverage"] = text_coverage(
                    src_text, "\n".join(chunks))
            full_text = "\n".join(chunks)
            post_type, _, _ = classify_content(full_text, os.path.basename(path))
            doc_type = mark_scan(post_type, metrics.get("ocr_confidence"))
            pages = metrics.get("num_pages", 0)
            cost = pages * ENGINE_RATES.get(engine, 0)
            duration = time.monotonic() - started
            kb.record_telemetry(doc.doc_id, wave, doc_type, engine, metrics,
                                cost, duration, status="ok")
            if self.force:
                kb.add_reparse_history(doc.doc_id, engine, before or {},
                                       kb.telemetry_row(doc.doc_id) or {}, cost)
            self._bump("done")
            return {"source": path, "status": "ok", "doc_id": doc.doc_id,
                    "doc_type": doc_type, "engine": engine,
                    "flags": red_flags(metrics)}
        except Exception as exc:  # noqa: BLE001 — everything lands somewhere
            kb.add_dead_letter(path, wave, f"{type(exc).__name__}: {exc}")
            self._bump("failed")
            return {"source": path, "status": "failed", "error": str(exc)}
        finally:
            kb.close()

    def run(self, paths: list[str], wave: str,
            progress_cb=None) -> dict:
        """Ingest paths as one named wave. Returns a summary dict."""
        kb = KnowledgeBase(self.kb_path)
        try:
            kb.create_wave(wave, len(paths))
        finally:
            kb.close()
        self.progress = {"done": 0, "failed": 0, "skipped": 0, "total": len(paths)}
        results = []
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._ingest_one, p, wave): p for p in paths}
                for fut in as_completed(futures):
                    res = fut.result()
                    results.append(res)
                    if progress_cb:
                        progress_cb(dict(self.progress), res)
        finally:
            kb = KnowledgeBase(self.kb_path)
            try:
                # Waves always "complete"; failures live in the dead-letter
                # queue rather than failing the wave.
                kb.complete_wave(wave, "done")
            finally:
                kb.close()
        summary = dict(self.progress)
        summary["results"] = results
        return summary
