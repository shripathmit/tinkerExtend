"""SQLite knowledge base: documents + chunks + FTS5 full-text search."""
from __future__ import annotations

import json
import re
import sqlite3

from .models import Chunk, Document


STOPWORDS = frozenset(
    "what is the a an was were are be been has have had does did do of in on for "
    "to and or by with as at from that this it its which who whom whose how much "
    "many when where why s t d no not there their they them then than so such "
    "into over under about between through during each other some any all both "
    "few more most only own same too very can will just should now".split()
)


def significant_terms(query: str) -> list[str]:
    """Alphanumeric terms worth matching: lowercase, stopwords removed."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in re.findall(r"[A-Za-z0-9]+", query.lower()):
        if len(tok) > 1 and tok not in STOPWORDS and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def sanitize_fts_query(query: str) -> str:
    """Make raw user text safe for FTS5 MATCH (OR of significant terms)."""
    return " OR ".join(significant_terms(query))

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
CREATE TABLE IF NOT EXISTS parse_telemetry (
  doc_id         TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
  wave           TEXT NOT NULL DEFAULT '',
  doc_type       TEXT NOT NULL DEFAULT '',
  engine         TEXT NOT NULL DEFAULT '',
  num_pages      INTEGER NOT NULL DEFAULT 0,
  num_chunks     INTEGER NOT NULL DEFAULT 0,
  num_tables     INTEGER NOT NULL DEFAULT 0,
  empty_chunks   INTEGER NOT NULL DEFAULT 0,
  garbled_chunks INTEGER NOT NULL DEFAULT 0,
  text_coverage  REAL,
  ocr_confidence REAL,
  table_shape_ok INTEGER NOT NULL DEFAULT 0,
  repeated_ngrams INTEGER NOT NULL DEFAULT 0,
  cost_usd       REAL NOT NULL DEFAULT 0,
  duration_s     REAL NOT NULL DEFAULT 0,
  status         TEXT NOT NULL DEFAULT 'ok',   -- ok | failed | skipped
  error          TEXT NOT NULL DEFAULT '',
  created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dead_letter (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  source     TEXT NOT NULL,
  wave       TEXT NOT NULL DEFAULT '',
  attempts   INTEGER NOT NULL DEFAULT 0,
  last_error TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  resolved   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS waves (
  name         TEXT PRIMARY KEY,
  target       INTEGER NOT NULL DEFAULT 0,
  status       TEXT NOT NULL DEFAULT 'running',  -- running | done | failed
  created_at   TEXT NOT NULL,
  completed_at TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   TEXT NOT NULL,
  question     TEXT NOT NULL,
  answer_mode  TEXT NOT NULL DEFAULT '',
  answer       TEXT NOT NULL DEFAULT '',
  chunk_ids    TEXT NOT NULL DEFAULT '[]',
  rating       INTEGER NOT NULL,          -- +1 helpful, -1 not helpful
  comment      TEXT NOT NULL DEFAULT ''
);
"""


class KnowledgeBase:
    def __init__(self, path: str = "./kb.sqlite3"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Idempotent column additions for databases created before them."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(documents)")}
        if "doc_type" not in cols:
            self.conn.execute("ALTER TABLE documents ADD COLUMN doc_type TEXT NOT NULL DEFAULT ''")
            self.conn.commit()

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

    def _fts(self, fts_query: str, limit: int) -> list[dict]:
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
            (fts_query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Strict AND over significant terms first; OR fallback if empty.

        AND keeps unanswerable questions answerless (so the QA layer can
        abstain); the OR fallback keeps natural phrasing working.
        """
        terms = significant_terms(query)
        if not terms:
            return []
        hits = self._fts(" AND ".join(terms), limit)
        if not hits:
            hits = self._fts(" OR ".join(terms), limit)
        return hits

    def stats(self) -> dict:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        tokens = self.conn.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM chunks"
        ).fetchone()[0]
        return {"documents": docs, "chunks": chunks, "token_estimate": tokens}

    def add_feedback(self, question: str, answer_mode: str, answer: str,
                     chunk_ids: list, rating: int, comment: str = "") -> int:
        """Record a thumbs up/down on an answer. Returns the feedback id."""
        import datetime
        cur = self.conn.execute(
            "INSERT INTO feedback (created_at, question, answer_mode, answer,"
            " chunk_ids, rating, comment) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.datetime.now(datetime.timezone.utc).isoformat(),
             question, answer_mode, answer[:4000],
             json.dumps(chunk_ids or []), 1 if rating > 0 else -1,
             comment[:2000]),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_feedback(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Phase 0/1: telemetry, waves, dead letters -----------------------
    @staticmethod
    def _now() -> str:
        import datetime
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def record_telemetry(self, doc_id: str, wave: str, doc_type: str,
                         engine: str, metrics: dict, cost_usd: float,
                         duration_s: float, status: str = "ok",
                         error: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO parse_telemetry (doc_id, wave, doc_type,"
            " engine, num_pages, num_chunks, num_tables, empty_chunks,"
            " garbled_chunks, text_coverage, ocr_confidence, table_shape_ok,"
            " repeated_ngrams, cost_usd, duration_s, status, error, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (doc_id, wave, doc_type, engine,
             metrics.get("num_pages", 0), metrics.get("num_chunks", 0),
             metrics.get("table_blocks", 0), metrics.get("empty_chunks", 0),
             metrics.get("garbled_chunks", 0), metrics.get("text_coverage"),
             metrics.get("ocr_confidence"),
             metrics.get("table_shape_ok", 0),
             metrics.get("repeated_ngrams", 0),
             cost_usd, duration_s, status, error[:2000], self._now()),
        )
        if doc_type:
            self.conn.execute("UPDATE documents SET doc_type = ? WHERE doc_id = ?",
                              (doc_type, doc_id))
        self.conn.commit()

    def update_telemetry_metrics(self, doc_id: str, metrics: dict) -> bool:
        """Recompute fidelity metrics for a doc; keeps wave/engine/cost/duration."""
        cur = self.conn.execute(
            "UPDATE parse_telemetry SET num_pages=?, num_chunks=?, num_tables=?,"
            " empty_chunks=?, garbled_chunks=?, text_coverage=?, ocr_confidence=?,"
            " table_shape_ok=?, repeated_ngrams=? WHERE doc_id=?",
            (metrics.get("num_pages", 0), metrics.get("num_chunks", 0),
             metrics.get("table_blocks", 0), metrics.get("empty_chunks", 0),
             metrics.get("garbled_chunks", 0), metrics.get("text_coverage"),
             metrics.get("ocr_confidence"), metrics.get("table_shape_ok", 0),
             metrics.get("repeated_ngrams", 0), doc_id))
        self.conn.commit()
        return cur.rowcount > 0

    def telemetry(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT t.*, d.source FROM parse_telemetry t"
            " JOIN documents d ON d.doc_id = t.doc_id"
            " ORDER BY t.created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def fidelity_by_type(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT doc_type, COUNT(*) AS docs,"
            " AVG(CASE WHEN num_tables>0 THEN CAST(table_shape_ok AS REAL)/num_tables ELSE 1 END) AS table_shape_rate,"
            " AVG(CAST(empty_chunks AS REAL)/NULLIF(num_chunks,0)) AS empty_rate,"
            " AVG(CAST(garbled_chunks AS REAL)/NULLIF(num_chunks,0)) AS garbled_rate,"
            " AVG(ocr_confidence) AS ocr_confidence,"
            " AVG(text_coverage) AS text_coverage,"
            " SUM(cost_usd) AS cost_usd"
            " FROM parse_telemetry WHERE status='ok' GROUP BY doc_type").fetchall()
        return [dict(r) for r in rows]

    def overview(self) -> dict:
        t = self.conn.execute(
            "SELECT COUNT(*) AS docs, COALESCE(SUM(cost_usd),0) AS cost,"
            " SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok,"
            " SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed"
            " FROM parse_telemetry").fetchone()
        dl = self.conn.execute(
            "SELECT COUNT(*) FROM dead_letter WHERE resolved=0").fetchone()[0]
        fb = self.conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN rating>0 THEN 1 ELSE 0 END)"
            " FROM feedback").fetchone()
        return {
            "docs": t["docs"], "cost_usd": round(t["cost"], 2),
            "ok": t["ok"], "failed": t["failed"],
            "success_rate": (t["ok"] / t["docs"]) if t["docs"] else 1.0,
            "dead_letters": dl,
            "feedback_total": fb[0], "feedback_positive": fb[1] or 0,
        }

    def add_dead_letter(self, source: str, wave: str, error: str) -> None:
        row = self.conn.execute(
            "SELECT id, attempts FROM dead_letter WHERE source=? AND resolved=0",
            (source,)).fetchone()
        if row:
            self.conn.execute(
                "UPDATE dead_letter SET attempts=attempts+1, last_error=?,"
                " wave=?, created_at=? WHERE id=?",
                (error[:2000], wave, self._now(), row["id"]))
        else:
            self.conn.execute(
                "INSERT INTO dead_letter (source, wave, attempts, last_error,"
                " created_at) VALUES (?,?,?,?,?)",
                (source, wave, 1, error[:2000], self._now()))
        self.conn.commit()

    def list_dead_letters(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM dead_letter WHERE resolved=0 ORDER BY id DESC").fetchall()]

    def resolve_dead_letter(self, dl_id: int) -> None:
        self.conn.execute("UPDATE dead_letter SET resolved=1 WHERE id=?", (dl_id,))
        self.conn.commit()

    def requeue_dead_letter(self, dl_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM dead_letter WHERE id=? AND resolved=0", (dl_id,)).fetchone()
        if not row:
            return None
        self.conn.execute("UPDATE dead_letter SET resolved=1 WHERE id=?", (dl_id,))
        self.conn.commit()
        return dict(row)

    def create_wave(self, name: str, target: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO waves (name, target, status, created_at)"
            " VALUES (?,?, 'running', ?)", (name, target, self._now()))
        self.conn.commit()

    def complete_wave(self, name: str, status: str = "done") -> None:
        self.conn.execute(
            "UPDATE waves SET status=?, completed_at=? WHERE name=?",
            (status, self._now(), name))
        self.conn.commit()

    def waves(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT w.*, "
            "(SELECT COUNT(*) FROM parse_telemetry t WHERE t.wave=w.name) AS ingested,"
            "(SELECT COUNT(*) FROM parse_telemetry t WHERE t.wave=w.name AND t.status='ok') AS ok,"
            "(SELECT COUNT(*) FROM parse_telemetry t WHERE t.wave=w.name AND t.status='failed') AS failed,"
            "(SELECT COALESCE(SUM(cost_usd),0) FROM parse_telemetry t WHERE t.wave=w.name) AS cost_usd"
            " FROM waves w ORDER BY w.created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None
