import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tinker_extend.models import Chunk, Document
from tinker_extend.store import KnowledgeBase


def make_kb():
    return KnowledgeBase(":memory:")


def sample_doc():
    return Document(doc_id="hash1", source="/tmp/a.pdf", file_hash="hash1",
                    parse_run_id="pr_1", num_pages=2)


def sample_chunks():
    return [
        Chunk(chunk_id="hash1:00000", doc_id="hash1", ordinal=0,
              content="The refund policy allows returns within thirty days of purchase.",
              pages_json="[1]", token_estimate=20),
        Chunk(chunk_id="hash1:00001", doc_id="hash1", ordinal=1,
              content="Shipping takes two business days for domestic orders.",
              pages_json="[2]", token_estimate=18),
    ]


def test_ingest_and_search():
    kb = make_kb()
    kb.upsert_document(sample_doc())
    kb.add_chunks(sample_chunks())

    assert kb.has_hash("hash1")
    assert not kb.has_hash("nope")

    hits = kb.search("refund policy")
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "hash1:00000"
    assert hits[0]["doc_source"] == "/tmp/a.pdf"
    assert "refund" in hits[0]["snippet"].lower()

    hits = kb.search("shipping")
    assert len(hits) == 1 and hits[0]["ordinal"] == 1

    assert kb.search("zebras in antarctica") == []

    stats = kb.stats()
    assert stats == {"documents": 1, "chunks": 2, "token_estimate": 38,
                     "embedded_chunks": 0}
    kb.close()


def test_reingest_replaces_chunks():
    kb = make_kb()
    kb.upsert_document(sample_doc())
    kb.add_chunks(sample_chunks())
    # re-ingest same doc with different content (pipeline uses replace_chunks)
    kb.upsert_document(sample_doc())
    kb.replace_chunks("hash1", [Chunk(chunk_id="hash1:00000", doc_id="hash1", ordinal=0,
                                      content="Completely new content about warranties.",
                                      token_estimate=10)])
    assert kb.search("refund policy") == []
    assert len(kb.search("warranties")) == 1
    assert kb.stats()["chunks"] == 1
    kb.close()


def test_delete_document():
    kb = make_kb()
    kb.upsert_document(sample_doc())
    kb.add_chunks(sample_chunks())
    kb.delete_document("hash1")
    assert kb.search("refund") == []
    assert kb.stats()["documents"] == 0
    kb.close()


def test_search_natural_language_query():
    from tinker_extend.store import sanitize_fts_query, significant_terms
    kb = make_kb()
    kb.upsert_document(sample_doc())
    kb.add_chunks(sample_chunks())
    assert significant_terms("What was total revenue?") == ["total", "revenue"]
    assert sanitize_fts_query("!!!") == ""
    hits = kb.search("What is the refund policy?")
    assert hits and hits[0]["chunk_id"] == "hash1:00000"
    assert kb.search("What is the CEO's hobby?") == []  # AND finds nothing, OR neither
    assert kb.search("???") == []
    kb.close()
