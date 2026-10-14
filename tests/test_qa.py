"""Tests for Q&A answer modes and shared page counting."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tinker_extend.chunking import chunks_from_parse_run, num_pages_in_run
from tinker_extend.fidelity import fidelity_metrics
from tinker_extend.models import Chunk, Document
from tinker_extend.qa import ask
from tinker_extend.store import KnowledgeBase


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def chat(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return "R&D fell by $207 million [1]."


def _kb():
    kb = KnowledgeBase(":memory:")
    kb.upsert_document(Document(doc_id="d1", source="/tmp/10q.pdf", file_hash="d1"))
    kb.replace_chunks("d1", [
        Chunk(chunk_id="d1:00000", doc_id="d1", ordinal=0,
              content="Research and development expenses decreased by $207 million.",
              pages_json="[4]"),
    ])
    return kb


def test_generated_answer_builds_prompt_with_pages():
    llm = FakeLLM()
    result = ask("How much did research and development expenses decrease?",
                 _kb(), llm=llm)
    assert result["mode"] == "generated"
    assert result["answer"] == "R&D fell by $207 million [1]."
    assert "[1] (source: /tmp/10q.pdf, pages [4])" in llm.prompts[0]


def test_page_count_agrees_between_chunking_and_fidelity():
    # Mixed shapes: nested metadata.page.number and the flat fallbacks.
    blocks = [
        {"type": "text", "content": "a", "metadata": {"page": {"number": 1}}},
        {"type": "text", "content": "b", "pageNumber": 2},
        {"type": "text", "content": "c", "page": 3},
    ]
    run = {"output": {"chunks": [{"content": "abc", "blocks": blocks}]}}
    chunks = chunks_from_parse_run(run, "d")
    assert num_pages_in_run(run) == 3
    assert fidelity_metrics(blocks, [c.content for c in chunks])["num_pages"] == 3
