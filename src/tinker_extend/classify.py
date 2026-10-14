"""Heuristic document-type classifier.

Phase 0 needs a doc-type taxonomy before any sampling or per-type
engine profiles make sense. This is deliberately rules-based and
explainable — no model, no API call — so the taxonomy stays stable
while the corpus grows. A filename pass picks the parse engine
(pre-parse); a content pass refines the label after parsing.
"""
from __future__ import annotations

import re

# (doc_type, [filename patterns], [content patterns], base confidence)
_RULES: list[tuple[str, list[str], list[str], float]] = [
    ("financial_filing",
     [r"10-?q", r"10-?k", r"8-?k", r"20-?f", r"s-?1\b", r"10q", r"10k"],
     [r"securities and exchange commission", r"form 10-q", r"form 10-k",
      r"quarterly report pursuant to section 13"],
     0.9),
    ("contract",
     [r"agreement", r"contract", r"\bmsa\b", r"\bsow\b", r"\bnda\b", r"amendment"],
     [r"this agreement", r"whereas", r"indemnif", r"governing law",
      r"counterparts"],
     0.8),
    ("invoice",
     [r"invoice", r"bill", r"receipt"],
     [r"invoice number", r"amount due", r"billing address", r"purchase order"],
     0.85),
    ("email",
     [r"\.eml$", r"\.msg$", r"email"],
     [r"\bfrom:.*\bto:", r"subject:", r"sent:"],
     0.85),
    ("report",
     [r"report", r"memo", r"whitepaper", r"brief"],
     [r"executive summary", r"memorandum", r"key findings"],
     0.7),
    ("presentation",
     [r"\.pptx?$", r"deck", r"slides", r"presentation", r"earnings"],
     [r"slide \d+", r"agenda"],
     0.8),
]


def classify_filename(filename: str) -> tuple[str, float, list[str]]:
    """Pre-parse classification from the filename alone (picks the engine)."""
    name = (filename or "").lower()
    for doc_type, fn_pats, _, conf in _RULES:
        hits = [p for p in fn_pats if re.search(p, name)]
        if hits:
            return doc_type, conf, [f"filename matched {hits[0]}"]
    return "other", 0.4, ["no filename rule matched"]


def classify_content(text: str, filename: str = "") -> tuple[str, float, list[str]]:
    """Post-parse classification, refining the filename guess with content."""
    sample = (text or "").lower()[:20000]
    best: tuple[str, float, list[str]] = ("other", 0.4, ["no content rule matched"])
    for doc_type, fn_pats, content_pats, conf in _RULES:
        c_hits = [p for p in content_pats if re.search(p, sample)]
        if not c_hits:
            continue
        fn_hit = any(re.search(p, (filename or "").lower()) for p in fn_pats)
        score = conf + (0.05 if fn_hit else 0.0)
        reasons = [f"content matched {c_hits[0]}"] + (
            ["filename corroborates"] if fn_hit else [])
        if score > best[1]:
            best = (doc_type, min(score, 0.99), reasons)
    # Scanned-document override is applied by the pipeline from OCR confidence.
    return best


def mark_scan(doc_type: str, ocr_confidence: float | None) -> str:
    """Low OCR confidence means the doc is effectively a scan."""
    if ocr_confidence is not None and ocr_confidence < 0.85:
        return "scan"
    return doc_type


DOC_TYPES = [r[0] for r in _RULES] + ["scan", "other"]
