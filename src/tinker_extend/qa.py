"""Question answering over the knowledge base.

Two modes:
  extractive - top passages returned as the answer (no LLM needed, no hallucination)
  generated  - an LLM synthesizes an answer grounded ONLY in the passages,
               with [n] citations (needs TINKER_LLM_API_KEY)
"""
from __future__ import annotations

import os

from .embeddings import Embedder
from .llm import LLMClient
from .store import KnowledgeBase, significant_terms

# Cosine similarity at which a vector match alone is relevant enough to
# answer from. bge-small scores related text ~0.7-0.9; tune with the eval.
DEFAULT_MIN_SIM = 0.80


def min_similarity() -> float:
    try:
        return float(os.environ.get("TINKER_MIN_SIM", DEFAULT_MIN_SIM))
    except ValueError:
        return DEFAULT_MIN_SIM


SYSTEM_PROMPT = """You answer questions using ONLY the passages below.
Rules:
- Every factual claim must be supported by a passage; cite it as [1], [2], etc.
- If the passages do not contain the answer, say "I couldn't find that in the documents."
- Be concise. Prefer numbers and direct quotes from the passages."""


def _passage_block(i: int, p: dict) -> str:
    return f"[{i}] (source: {p['doc_source']}, pages {p['pages']})\n{p['content']}"


def retrieve(question: str, kb: KnowledgeBase, embedder: Embedder | None = None,
             limit: int = 5) -> list[dict]:
    """Hybrid search when an embedder with a model is given, else keyword."""
    if embedder is not None and embedder.model:
        qvec = embedder.embed_query(question)
        if qvec is not None:
            return kb.search(question, limit=limit, query_vec=qvec,
                             model=embedder.model)
    return kb.search(question, limit=limit)


def ask(question: str, kb: KnowledgeBase, llm: LLMClient | None = None,
        limit: int = 5, embedder: Embedder | None = None,
        min_sim: float | None = None) -> dict:
    hits = retrieve(question, kb, embedder=embedder, limit=limit)
    passages = [
        {
            "n": i + 1,
            "chunk_id": h["chunk_id"],
            "doc_source": h["doc_source"],
            "pages": h["pages_json"],
            "content": h["content"],
            "match": h.get("match", "keyword"),
            "vector_score": h.get("vector_score"),
        }
        for i, h in enumerate(hits)
    ]
    terms = significant_terms(question)
    no_answer = {
        "question": question,
        "mode": "none",
        "answer": "I couldn't find anything relevant in the knowledge base.",
        "passages": [],
    }
    if not passages:
        return no_answer
    # Relevance gate: the top passage must share at least min(2, n_terms)
    # significant terms with the question, or (hybrid search) be a close
    # vector match, or we abstain instead of answering from a coincidental
    # single-word overlap.
    top = passages[0]
    overlap = sum(1 for t in terms if t in top["content"].lower())
    threshold = min_similarity() if min_sim is None else min_sim
    close_vector = (top["vector_score"] or 0.0) >= threshold
    if overlap < min(2, len(terms)) and not close_vector:
        return no_answer
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
