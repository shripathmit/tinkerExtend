import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tinker_extend.eval import (
    EvalCase, eval_answers, eval_retrieval, load_eval_set, parse_health,
)
from tinker_extend.models import Chunk, Document
from tinker_extend.store import KnowledgeBase


def make_kb():
    kb = KnowledgeBase(":memory:")
    kb.upsert_document(Document(doc_id="d1", source="/tmp/a.pdf", file_hash="d1"))
    kb.add_chunks([
        Chunk(chunk_id="d1:00000", doc_id="d1", ordinal=0,
              content="The refund policy allows returns within thirty days.",
              token_estimate=15),
        Chunk(chunk_id="d1:00001", doc_id="d1", ordinal=1,
              content="Shipping takes two business days.", token_estimate=10),
    ])
    return kb


CASES = [
    EvalCase(question="What is the refund policy?",
             relevant_contains=["refund policy"],
             answer_contains=["thirty days"], answerable=True),
    EvalCase(question="What is the CEO's hobby?",
             relevant_contains=[], answer_contains=[], answerable=False),
]


def test_eval_retrieval():
    kb = make_kb()
    rep = eval_retrieval(kb, CASES, k=5)
    assert rep["n"] == 1  # unanswerable cases are excluded from retrieval metrics
    assert rep["hit_rate"] == 1.0
    assert rep["per_case"][0]["hit"] and rep["per_case"][0]["rank"] == 1
    assert rep["mrr"] == 1.0
    kb.close()


def test_relevance_gate_abstains_on_coincidental_overlap():
    from tinker_extend.qa import ask
    kb = KnowledgeBase(":memory:")
    kb.upsert_document(Document(doc_id="d9", source="/tmp/fda.pdf", file_hash="d9"))
    kb.add_chunks([Chunk(chunk_id="d9:00000", doc_id="d9", ordinal=0,
                         content="The Food and Drug Administration regulates biologics.",
                         token_estimate=12)])
    # only "food" overlaps -> must abstain, not answer about the FDA
    result = ask("What is the CEO's favorite food?", kb)
    assert result["mode"] == "none"
    # single-term question still works
    result = ask("biologics", kb)
    assert result["mode"] == "extractive"
    kb.close()


def test_eval_answers_extractive():
    kb = make_kb()
    rep = eval_answers(kb, CASES, llm=None, k=5)
    assert rep["per_case"][0]["score"] == 1.0  # "thirty days" in top chunk
    assert rep["per_case"][1]["abstained"] is True
    assert rep["mean_score"] == 1.0
    kb.close()


def test_parse_health():
    kb = make_kb()
    h = parse_health(kb)
    assert h["kb_stats"]["documents"] == 1
    assert h["documents"][0]["chunks"] == 2
    kb.close()


def test_load_eval_set(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text('{"question": "q?", "relevant_contains": ["a"], "answer_contains": ["b"]}\n')
    cases = load_eval_set(str(p))
    assert len(cases) == 1 and cases[0].answerable is True
