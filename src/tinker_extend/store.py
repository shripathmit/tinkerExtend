"""SQLite knowledge base: documents + chunks + FTS5 full-text search."""
from __future__ import annotations

import json
import sqlite3

from .models import Chunk, Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id       TEXT PRIMARY KEY,
  source       TEXT NOT NULL,
  file_hash    TEXT NOT NULL UNIQUE,
  extend_file_id TEXT,
  parse_run_id TEXT,
  status       TEXT NOT NULL DEFAULT 'ingested',
  num_pages    INTEGER NOT NULL DEFAULT 0,
  num_chunks   INTEGER NOT NULL DEFAULT 0,
  ingested_at  TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id       TEXT PRIMARY KEY,
  doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  ordinal        INTEGER NOT NULL,
  content        TEXT NOT NULL,
  blocks_json    TEXT NOT NULL DEFAULT '[]',
  pages_json     TEXT NOT NULL DEFAULT '[]',
  token_estimate INTEGER NOT NULL DEFAULT 0,
  embedding_json TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id, content);
"""


class KnowledgeBase:
    def __init__(self, path: str = "./kb.sqlite3"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- writes ----------------------------------------------------------
    def upsert_document(self, doc: Document) -> None:
        self.conn.execute(
            """INSERT INTO documents
               (doc_id, source, file_hash, extend_file_id, parse_run_id, status,
                num_pages, num_chunks, ingested_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 source=excluded.source, extend_file_id=excluded.extend_file_id,
                 parse_run_id=excluded.parse_run_id, status=excluded.status,
                 num_pages=excluded.num_pages, num_chunks=excluded.num_chunks,
                 ingested_at=excluded.ingested_at, metadata_json=excluded.metadata_json""",
            (
                doc.doc_id, doc.source, doc.file_hash, doc.extend_file_id,
                doc.parse_run_id, doc.status, doc.num_pages, doc.num_chunks,
                doc.ingested_at, doc.metadata_json,
            ),
        )
        self.conn.commit()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        with self.conn:
            for c in chunks:
                self.conn.execute(
                    """INSERT INTO chunks
                       (chunk_id, doc_id, ordinal, content, blocks_json, pages_json,
                        token_estimate, embedding_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         content=excluded.content, blocks_json=excluded.blocks_json,
                         pages_json=excluded.pages_json,
                         token_estimate=excluded.token_estimate,
                         embedding_json=excluded.embedding_json""",
                    (
                        c.chunk_id, c.doc_id, c.ordinal, c.content, c.blocks_json,
                        c.pages_json, c.token_estimate, c.embedding_json,
                    ),
                )
                # FTS5 has no UPSERT: delete then re-insert.
                self.conn.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?", (c.chunk_id,)
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, content) VALUES (?, ?)",
                    (c.chunk_id, c.content),
                )
            if chunks:
                self.conn.execute(
                    "UPDATE documents SET num_chunks = ? WHERE doc_id = ?",
                    (len(chunks), chunks[0].doc_id),
                )

    def replace_chunks(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Replace all chunks of a document (used on re-ingest)."""
        with self.conn:
            old_ids = [
                r[0] for r in self.conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
                )
            ]
            for cid in old_ids:
                self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.add_chunks(chunks)

    def delete_document(self, doc_id: str) -> None:
        with self.conn:
            chunk_ids = [
                r[0] for r in self.conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
                )
            ]
            for cid in chunk_ids:
                self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    # -- reads -----------------------------------------------------------
    def has_hash(self, file_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return row is not None

    def search(self, query: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            """SELECT c.chunk_id, c.doc_id, c.ordinal, c.content,
                      c.pages_json, c.token_estimate,
                      d.source AS doc_source,
                      snippet(chunks_fts, 1, '<b>', '</b>', '…', 24) AS snippet,
                      rank AS fts_rank
               FROM chunks_fts
               JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
               JOIN documents d ON d.doc_id = c.doc_id
               WHERE chunks_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        tokens = self.conn.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM chunks"
        ).fetchone()[0]
        return {"documents": docs, "chunks": chunks, "token_estimate": tokens}

    def get_document(self, doc_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None
