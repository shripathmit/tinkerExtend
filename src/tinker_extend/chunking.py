"""Normalize Extend parse output into knowledge-base chunk records."""
from __future__ import annotations

import json

from .models import Chunk


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English prose)."""
    return max(1, len(text) // 4)


def block_page_number(b) -> int | None:
    """Page number of one layout block, or None.

    Real Extend blocks nest the page under metadata.page.number; keep the
    flat fallbacks for older/other shapes. Shared by chunking and fidelity
    so both count pages the same way.
    """
    if not isinstance(b, dict):
        return None
    meta_page = (b.get("metadata") or {}).get("page") or {}
    for c in (meta_page.get("number"), b.get("page"),
              b.get("pageNumber"), b.get("page_number")):
        if isinstance(c, int):
            return c
    return None


def _pages_from_blocks(blocks: list) -> list[int]:
    pages = {p for p in map(block_page_number, blocks) if p is not None}
    return sorted(pages)


def chunks_from_parse_run(run: dict, doc_id: str) -> list[Chunk]:
    """Turn an Extend parse_run's output.chunks into Chunk records."""
    output = run.get("output") or {}
    raw_chunks = output.get("chunks") or []
    chunks: list[Chunk] = []
    for i, raw in enumerate(raw_chunks):
        content = (raw.get("content") or "").strip()
        if not content:
            continue
        blocks = raw.get("blocks") or []
        chunk = Chunk(
            chunk_id=f"{doc_id}:{i:05d}",
            doc_id=doc_id,
            ordinal=i,
            content=content,
            blocks_json=json.dumps(blocks),
            pages_json=json.dumps(_pages_from_blocks(blocks)),
            token_estimate=estimate_tokens(content),
        )
        chunks.append(chunk)
    return chunks


def num_pages_in_run(run: dict) -> int:
    output = run.get("output") or {}
    raw_chunks = output.get("chunks") or []
    pages: set[int] = set()
    for raw in raw_chunks:
        pages.update(_pages_from_blocks(raw.get("blocks") or []))
    return len(pages)


def re_chunk_text(text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> list[str]:
    """Split text into overlapping windows (word-based; ~4 chars ≈ 1 token)."""
    words = text.split()
    if not words:
        return []
    max_words = max(1, int(max_tokens * 0.75))  # ~0.75 words per token
    overlap_words = max(0, int(overlap_tokens * 0.75))
    out: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        out.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return out
