"""Question answering over the knowledge base.

Two modes:
  extractive - top passages returned as the answer (no LLM needed, no hallucination)
  generated  - an LLM synthesizes an answer grounded ONLY in the passages,
               with [n] citations (needs TINKER_LLM_API_KEY)
"""
from __future__ import annotations

from .llm import LLMClient
from .store import KnowledgeBase

SYSTEM_PROMPT = """You answer questions using ONLY the passages below.
Rules:
- Every factual claim must be supported by a passage; cite it as [1], [2], etc.
- If the passages do not contain the answer, say "I couldn't find that in the documents."
- Be concise. Prefer numbers and direct quotes from the passages."""


def _passage_block(i: int, p: dict) -> str:
    return f"[{i}] (source: {p['doc_source']}, pages {p['pages_json']})\n{p['content']}"


def ask(question: str, kb: KnowledgeBase, llm: LLMClient | None = None,
        limit: int = 5) -> dict:
    hits = kb.search(question, limit=limit)
    passages = [
        {
            "n": i + 1,
            "chunk_id": h["chunk_id"],
            "doc_source": h["doc_source"],
            "pages": h["pages_json"],
            "content": h["content"],
        }
        for i, h in enumerate(hits)
    ]
    if not passages:
        return {
            "question": question,
            "mode": "none",
            "answer": "I couldn't find anything relevant in the knowledge base.",
            "passages": [],
        }
    if llm is None:
        top = passages[0]["content"]
        return {
            "question": question,
            "mode": "extractive",
            "answer": top[:1500],
            "note": "Extractive answer (no LLM configured). Set TINKER_LLM_API_KEY for generated answers.",
            "passages": passages,
        }
    user_prompt = (
        f"Question: {question}\n\nPassages:\n"
        + "\n\n".join(_passage_block(p["n"], p) for p in passages)
    )
    answer = llm.chat(SYSTEM_PROMPT, user_prompt)
    return {
        "question": question,
        "mode": "generated",
        "answer": answer,
        "passages": passages,
    }
