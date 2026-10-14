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

from .qa import ask, retrieve
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


def eval_retrieval(kb: KnowledgeBase, cases: list[EvalCase], k: int = 5,
                   embedder=None) -> dict:
    """hit_rate@k and MRR over the answerable golden cases.

    Unanswerable cases are excluded here — retrieving nothing for them is
    correct, and that is measured by eval_answers' abstention check.
    """
    per_case = []
    for case in cases:
        if not case.answerable:
            continue
        hits = retrieve(case.question, kb, embedder=embedder, limit=k)
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


def eval_answers(kb: KnowledgeBase, cases: list[EvalCase], llm=None, k: int = 5,
                 embedder=None) -> dict:
    """Answer quality: expected facts present; unanswerables abstain."""
    per_case = []
    for case in cases:
        result = ask(case.question, kb, llm=llm, limit=k, embedder=embedder)
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


def run_eval(kb: KnowledgeBase, cases: list[EvalCase], llm=None, k: int = 5,
             embedder=None) -> dict:
    return {
        "parse_health": parse_health(kb),
        "retrieval": eval_retrieval(kb, cases, k=k, embedder=embedder),
        "answers": eval_answers(kb, cases, llm=llm, k=k, embedder=embedder),
    }


def compare_modes(kb: KnowledgeBase, cases: list[EvalCase], embedder,
                  k: int = 5) -> dict:
    """Keyword-only vs hybrid on the same golden set, without per-case detail.

    Hybrid is reported as None when the embedder has no model (disabled).
    """
    def summary(emb):
        ret = eval_retrieval(kb, cases, k=k, embedder=emb)
        ans = eval_answers(kb, cases, llm=None, k=k, embedder=emb)
        return {"hit_rate": ret["hit_rate"], "mrr": ret["mrr"],
                "misses": ret["misses"], "answer_mean": ans["mean_score"],
                "abstention_rate": ans["abstention_rate"]}

    return {
        "k": k,
        "cases": len(cases),
        "keyword": summary(None),
        "hybrid": summary(embedder) if embedder is not None and embedder.model else None,
        "model": getattr(embedder, "model", None),
    }


def _where(hit: dict | None) -> str:
    """'file.pdf · p.4' for a search hit, '—' for none."""
    if not hit:
        return "—"
    try:
        pages = json.loads(hit.get("pages_json") or "[]")
    except (json.JSONDecodeError, TypeError):
        pages = []
    name = (hit.get("doc_source") or "").split("/")[-1]
    return f"{name} · p.{pages[0]}" if pages else name


def golden_assertions(kb: KnowledgeBase, cases: list[EvalCase], doc_checks: list[dict],
                      feedback: list[dict], embedder=None, k: int = 5) -> dict:
    """Every golden assertion as one row with a PASS/FAIL/RESOLVED status.

    Three sources:
      - golden Q&A cases: relevant passage in top-k, expected facts in the
        answer; unanswerable cases must be declined.
      - golden-doc probes (``doc_checks`` = check_golden_doc results, or
        ``{"doc", "status": "not_ingested"}``): each fact present in the parse.
      - Q&A feedback: a thumbs-up asserts the rated chunk is still retrieved
        in the top k for that question; a thumbs-down fails until resolved.
    """
    rows: list[dict] = []
    for case in cases:
        hits = retrieve(case.question, kb, embedder=embedder, limit=k)
        result = ask(case.question, kb, llm=None, limit=k, embedder=embedder)
        if not case.answerable:
            declined = result.get("mode") == "none"
            rows.append({
                "source": "golden_qa", "query": case.question,
                "expected": "Declines (unanswerable)",
                "retrieved": "Declined" if declined else _where(hits[0] if hits else None),
                "note": "Must not answer from unrelated passages",
                "status": "PASS" if declined else "FAIL"})
            continue
        rank = next((i for i, h in enumerate(hits, start=1)
                     if _contains_any(h["content"], case.relevant_contains)), None)
        answer = (result.get("answer") or "").lower()
        found = [f for f in case.answer_contains if f.lower() in answer]
        ok = rank is not None and len(found) == len(case.answer_contains)
        rows.append({
            "source": "golden_qa", "query": case.question,
            "expected": " | ".join(case.relevant_contains),
            "retrieved": (f"{_where(hits[rank - 1])} [rank {rank}]" if rank
                          else f"Not in top {k}" + (f" (top: {_where(hits[0])})" if hits else "")),
            "note": f"Facts {len(found)}/{len(case.answer_contains)} in answer",
            "status": "PASS" if ok else "FAIL"})

    for check in doc_checks:
        if check.get("status") == "not_ingested":
            rows.append({"source": "golden_doc", "query": f"Parse of {check.get('doc', '')}",
                         "expected": "Document ingested", "retrieved": "Not ingested",
                         "note": "Ingest the golden document to run its probes",
                         "status": "FAIL"})
            continue
        for probe in check.get("probes", []):
            rows.append({
                "source": "golden_doc", "query": probe["fact"],
                "expected": f"In parse of {check.get('doc', '')}",
                "retrieved": "Found" if probe["passed"] else "Missing: " + ", ".join(probe["missing"]),
                "note": "Parse fidelity probe",
                "status": "PASS" if probe["passed"] else "FAIL"})

    for fb in feedback:
        try:
            chunk_ids = json.loads(fb.get("chunk_ids") or "[]")
        except (json.JSONDecodeError, TypeError):
            chunk_ids = []
        target = fb.get("active_chunk_id") or (chunk_ids[0] if chunk_ids else "")
        base = {"source": "feedback", "id": fb["id"], "query": fb["question"],
                "created_at": fb.get("created_at", "")}
        if fb["rating"] < 0:
            note = fb.get("issue") or "Marked inaccurate"
            if fb.get("comment"):
                note += f": {fb['comment']}"
            rows.append({**base, "expected": f"Fix for chunk {target or '—'}",
                         "retrieved": "Flagged in Q&A", "note": note,
                         "status": "RESOLVED" if fb.get("resolved") else "FAIL"})
            continue
        hits = retrieve(fb["question"], kb, embedder=embedder, limit=k)
        rank = next((i for i, h in enumerate(hits, start=1) if h["chunk_id"] == target), None)
        exists = bool(target) and kb.conn.execute(
            "SELECT 1 FROM chunks WHERE chunk_id = ?", (target,)).fetchone() is not None
        rows.append({**base, "expected": f"Chunk {target or '—'} in top {k}",
                     "retrieved": (f"Rank {rank}" if rank else
                                   "Chunk no longer exists" if not exists else f"Not in top {k}"),
                     "note": "Verified accurate in Q&A",
                     "status": "PASS" if rank else "FAIL"})

    counted = [r for r in rows if r["status"] != "RESOLVED"]
    passed = sum(1 for r in counted if r["status"] == "PASS")
    return {
        "rows": rows,
        "total": len(counted),
        "passed": passed,
        "failed": len(counted) - passed,
        "pass_rate": passed / len(counted) if counted else 1.0,
    }
