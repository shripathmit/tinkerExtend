"""Plain data records flowing through the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Document:
    doc_id: str            # sha256 of file content (stable across re-ingests)
    source: str            # local path or URL the document came from
    file_hash: str         # same as doc_id; kept explicit for clarity
    extend_file_id: str | None = None
    parse_run_id: str | None = None
    status: str = "ingested"
    num_pages: int = 0
    num_chunks: int = 0
    ingested_at: str = field(default_factory=utcnow_iso)
    metadata_json: str = "{}"


@dataclass
class Chunk:
    chunk_id: str          # f"{doc_id}:{ordinal:05d}"
    doc_id: str
    ordinal: int
    content: str           # markdown text for this chunk
    blocks_json: str = "[]"   # typed layout blocks (text/table/figure/...) from Extend
    pages_json: str = "[]"    # page numbers this chunk spans
    token_estimate: int = 0
    embedding: bytes | None = None        # normalized float32 vector, when embedded
    embedding_model: str | None = None    # model that produced `embedding`
