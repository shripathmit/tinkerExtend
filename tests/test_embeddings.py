"""Tests for embeddings and hybrid (keyword + vector) search.

A deterministic fake embedder stands in for the real model: words are
mapped through a small synonym table, then hashed into a bag-of-words
vector, so paraphrases land close together without any download.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hashlib
import re

import numpy as np
import pytest

from tinker_extend import embeddings
from tinker_extend.embeddings import Embedder, NullEmbedder, normalize, to_blob
from tinker_extend.models import Chunk, Document
from tinker_extend.pipeline import KnowledgeBasePipeline, backfill_embeddings
from tinker_extend.qa import ask
from tinker_extend.store import KnowledgeBase

SYNONYMS = {
    "spending": "expenses", "costs": "expenses", "cost": "expenses",
    "fall": "decreased", "fell": "decreased", "drop": "decreased",
    "r": "research", "d": "development",
}
STOP = {"how", "much", "did", "the", "by", "and", "what", "was", "a", "of"}


class FakeEmbedder(Embedder):
    def __init__(self, model="fake-v1", dims=64):
        self.model = model
        self.dims = dims
        self.calls = 0

    def _vec(self, text):
        v = np.zeros(self.dims, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w in STOP:
                continue
            w = SYNONYMS.get(w, w)
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1.0
        return normalize(v)

    def embed(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]


RD = "Research and development expenses decreased by $207 million."
SHIP = "Shipping takes two business days for domestic orders."
REFUND = "The refund policy allows returns within thirty days of purchase."


def _kb(path=":memory:", embedder=None, texts=(RD, SHIP, REFUND)):
    kb = KnowledgeBase(path)
    kb.upsert_document(Document(doc_id="d1", source="/tmp/doc.pdf", file_hash="d1"))
    chunks = []
    for i, t in enumerate(texts):
        c = Chunk(chunk_id=f"d1:{i:05d}", doc_id="d1", ordinal=i, content=t,
                  pages_json=f"[{i + 1}]")
        if embedder is not None:
            c.embedding = to_blob(embedder.embed([t])[0])
            c.embedding_model = embedder.model
        chunks.append(c)
    kb.replace_chunks("d1", chunks)
    return kb


def test_search_without_query_vec_is_keyword_search():
    emb = FakeEmbedder()
    kb = _kb(embedder=emb)
    assert kb.search("refund policy") == kb.keyword_search("refund policy")
    assert "vector_score" not in kb.search("refund policy")[0]


def test_hybrid_finds_paraphrase_keyword_misses():
    emb = FakeEmbedder()
    kb = _kb(embedder=emb)
    q = "How much did R&D spending fall?"
    assert kb.keyword_search(q) == []
    hits = kb.search(q, query_vec=emb.embed_query(q), model=emb.model)
    assert hits[0]["content"] == RD
    assert hits[0]["match"] == "vector"
    assert hits[0]["vector_score"] > 0.8
    assert hits[0]["snippet"].startswith("Research and development")
    assert hits[0]["doc_source"] == "/tmp/doc.pdf"


def test_rrf_ranks_chunk_found_by_both_retrievers_first():
    emb = FakeEmbedder()
    kb = _kb(embedder=emb)
    q = "research expenses decreased"
    hits = kb.search(q, query_vec=emb.embed_query(q), model=emb.model)
    assert hits[0]["content"] == RD
    assert hits[0]["match"] == "both"
    assert hits[0]["rrf_score"] > hits[1]["rrf_score"]


def test_vectors_from_other_model_are_ignored():
    kb = _kb(embedder=FakeEmbedder(model="old-model"))
    emb = FakeEmbedder(model="new-model")
    q = "How much did R&D spending fall?"
    assert kb.search(q, query_vec=emb.embed_query(q), model=emb.model) == []
    assert kb.embedding_stats("new-model")["embedded"] == 0


def test_vector_cache_refreshes_after_writes(tmp_path):
    emb = FakeEmbedder()
    path = str(tmp_path / "kb.sqlite3")
    kb = _kb(path, embedder=emb, texts=(SHIP,))
    q = "How much did R&D spending fall?"
    qv = emb.embed_query(q)
    assert all(h["content"] != RD for h in kb.search(q, query_vec=qv, model=emb.model))
    rev = kb.vectors_rev()
    kb.close()

    kb = _kb(path, embedder=emb, texts=(SHIP, RD))  # re-ingest adds the R&D chunk
    assert kb.vectors_rev() > rev
    assert kb.search(q, query_vec=qv, model=emb.model)[0]["content"] == RD
    kb.close()


def test_backfill_embeds_missing_vectors_once():
    kb = _kb()  # stored without vectors, like a pre-embedding KB
    emb = FakeEmbedder()
    assert kb.embedding_stats(emb.model) == {"model": "fake-v1", "embedded": 0, "chunks": 3}
    assert backfill_embeddings(kb, emb, batch_size=2) == 3
    assert kb.embedding_stats(emb.model)["embedded"] == 3
    assert kb.stats()["embedded_chunks"] == 3
    assert backfill_embeddings(kb, emb) == 0
    # A model change makes every vector stale again.
    assert backfill_embeddings(kb, FakeEmbedder(model="fake-v2")) == 3


def test_backfill_with_null_embedder_is_a_noop():
    kb = _kb()
    assert backfill_embeddings(kb, NullEmbedder()) == 0


def test_pipeline_store_run_saves_vectors():
    emb = FakeEmbedder()
    kb = KnowledgeBase(":memory:")
    pipe = KnowledgeBasePipeline(client=None, kb=kb, embedder=emb)
    run = {"id": "pr_1", "output": {"chunks": [
        {"content": RD, "blocks": [{"metadata": {"page": {"number": 1}}}]}]}}
    pipe._store_run("h1", "/tmp/x.pdf", "h1", "file_1", run)
    row = kb.conn.execute("SELECT embedding, embedding_model FROM chunks").fetchone()
    assert row["embedding_model"] == "fake-v1"
    vec = np.frombuffer(row["embedding"], dtype=np.float32)
    assert vec.shape == (64,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    q = "How much did R&D spending fall?"
    assert pipe.query(q)[0]["content"] == RD


def test_answer_gate_uses_vector_similarity():
    emb = FakeEmbedder()
    kb = _kb(embedder=emb)
    q = "How much did R&D spending fall?"
    # Keyword-only: no overlap, so it declines.
    assert ask(q, kb)["mode"] == "none"
    # Hybrid: the close vector match is enough to answer.
    res = ask(q, kb, embedder=emb, min_sim=0.8)
    assert res["mode"] == "extractive"
    assert res["passages"][0]["match"] == "vector"
    # An unrelated question still declines even though vectors return something.
    assert ask("zebra migration routes in antarctica", kb, embedder=emb,
               min_sim=0.8)["mode"] == "none"


def test_embedder_from_env_respects_off(monkeypatch):
    monkeypatch.setattr(embeddings, "_SHARED", None)
    monkeypatch.setenv("TINKER_EMBEDDINGS", "off")
    emb = embeddings.embedder_from_env()
    assert isinstance(emb, NullEmbedder)
    assert emb.model is None and emb.embed_query("x") is None


@pytest.mark.skipif(os.environ.get("TINKER_TEST_REAL_EMBED") != "1",
                    reason="set TINKER_TEST_REAL_EMBED=1 to download and test the real model")
def test_real_fastembed_model():
    pytest.importorskip("fastembed")
    emb = embeddings.FastEmbedEmbedder(cache_dir=os.environ.get("TINKER_EMBED_CACHE"))
    try:
        emb._load()
    except Exception as exc:  # no network to the model host
        pytest.skip(f"model unavailable: {exc}")
    kb = _kb(embedder=emb)
    q = "How much did R&D spending fall?"
    hits = kb.search(q, query_vec=emb.embed_query(q), model=emb.model)
    assert hits[0]["content"] == RD
    assert len(emb.embed([RD])[0]) == 384
