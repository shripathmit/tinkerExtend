"""Evaluation framework: how good is the knowledge base, really?

Three layers (start here, grow as needed):

1. Parse health  — per-document metrics from the pipeline itself
   (parse status, chunks/doc, tables detected, empty-chunk rate). Cheap,
   runs on every ingest; see `parse_health()`.

2. Retrieval quality — a golden set of questions with known-relevant
   content. Measures hit_rate@k and MRR: does the KB surface the right
   chunks? This is the core "is my index any good" signal.

3. Answer quality — same golden set plus expected facts. For extractive
   answers: does the answer contain the expected fact? For generated
   answers: same check, plus abstention checks on unanswerable questions
   (the system MUST say it can't find those, not hallucinate).

Golden-set format (JSONL, one object per line):
  {"question": "...",
   "relevant_contains": ["substring expected in a relevant chunk"],
   "answer_contains": ["fact expected in the answer"],
   "answerable": true}
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .qa import ask
from .store import KnowledgeBase


@dataclass
class EvalCase:
    question: str
    relevant_contains: list[str]
    answer_contains: list[str]
    answerable: bool = True

    @staticmethod
    def from_dict(d: dict) -> "EvalCase":
        return EvalCase(
            question=d["question"],
            relevant_contains=d.get("relevant_contains", []),
            answer_contains=d.get("answer_contains", []),
            answerable=d.get("answerable", True),
        )


def load_eval_set(path: str) -> list[EvalCase]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


def _contains_any(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def eval_retrieval(kb: KnowledgeBase, cases: list[EvalCase], k: int = 5) -> dict:
    """hit_rate@k and MRR over the answerable golden cases.

    Unanswerable cases are excluded here — retrieving nothing for them is
    correct, and that is measured by eval_answers' abstention check.
    """
    per_case = []
    for case in cases:
        if not case.answerable:
            continue
        hits = kb.search(case.question, limit=k)
        rank = None
        for i, h in enumerate(hits, start=1):
            if _contains_any(h["content"], case.relevant_contains):
                rank = i
                break
        per_case.append({
            "question": case.question,
            "hit": rank is not None,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank if rank else 0.0,
        })
    n = len(per_case)
    return {
        "k": k,
        "n": n,
        "hit_rate": sum(1 for c in per_case if c["hit"]) / n if n else 0.0,
        "mrr": sum(c["reciprocal_rank"] for c in per_case) / n if n else 0.0,
        "misses": [c["question"] for c in per_case if not c["hit"]],
        "per_case": per_case,
    }


def eval_answers(kb: KnowledgeBase, cases: list[EvalCase], llm=None, k: int = 5) -> dict:
    """Answer quality: expected facts present; unanswerables abstain."""
    per_case = []
    for case in cases:
        result = ask(case.question, kb, llm=llm, limit=k)
        answer = result.get("answer", "")
        if not case.answerable:
            abstained = result.get("mode") == "none" or "couldn't find" in answer.lower()
            per_case.append({"question": case.question, "answerable": False,
                             "abstained": abstained, "score": 1.0 if abstained else 0.0})
            continue
        found = [n for n in case.answer_contains if n.lower() in answer.lower()]
        score = len(found) / len(case.answer_contains) if case.answer_contains else 1.0
        per_case.append({
            "question": case.question,
            "mode": result.get("mode"),
            "score": score,
            "facts_found": found,
            "facts_expected": case.answer_contains,
        })
    n = len(per_case)
    unanswerable = [c for c in per_case if not c.get("answerable", True)]
    return {
        "n": n,
        "mean_score": sum(c["score"] for c in per_case) / n if n else 0.0,
        "abstention_rate": (sum(1 for c in unanswerable if c["abstained"]) / len(unanswerable)
                            if unanswerable else 1.0),
        "per_case": per_case,
    }


def parse_health(kb: KnowledgeBase) -> dict:
    """Cheap ingest-health metrics straight from the KB metadata."""
    stats = kb.stats()
    docs = []
    rows = kb.conn.execute(
        "SELECT doc_id, source, status, num_pages, num_chunks, parse_run_id FROM documents"
    ).fetchall()
    for r in rows:
        d = dict(r)
        empty = kb.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ? AND LENGTH(TRIM(content)) < 20",
            (d["doc_id"],),
        ).fetchone()[0]
        tables = kb.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ? AND blocks_json LIKE '%\"table\"%'",
            (d["doc_id"],),
        ).fetchone()[0]
        docs.append({
            "source": d["source"].split("/")[-1],
            "status": d["status"],
            "pages": d["num_pages"],
            "chunks": d["num_chunks"],
            "empty_chunks": empty,
            "chunks_with_tables": tables,
        })
    return {"kb_stats": stats, "documents": docs}


def run_eval(kb: KnowledgeBase, cases: list[EvalCase], llm=None, k: int = 5) -> dict:
    return {
        "parse_health": parse_health(kb),
        "retrieval": eval_retrieval(kb, cases, k=k),
        "answers": eval_answers(kb, cases, llm=llm, k=k),
    }
