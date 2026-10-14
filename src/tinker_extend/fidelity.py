"""Parse-fidelity metrics (Phase 0): did we read the document correctly?

Two kinds of checks:

1. Intrinsic — computed for every document at ingest, no ground truth
   needed. These feed the admin dashboard and the red-flag list.
2. Golden documents — a handful of docs with hand-recorded probe facts
   and known table shapes, asserted on every run. Format lives in
   eval/golden_docs.json.

Thresholds in red_flags() are starting points; tune them against your
corpus once the 100-doc pilot wave lands.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

from .chunking import TAG_STRIP_RE as _TAG_STRIP_RE, block_page_number

# Bump when a metric's definition changes; the app recomputes stored
# metrics once at startup (from stored blocks, no API calls).
FIDELITY_VERSION = "2"

_REPLACEMENT = "\ufffd"
_LONG_TOKEN_RE = re.compile(r"\S{60,}")


def _is_garbled(text: str) -> bool:
    # Strip markup first: serialized tables legitimately contain very long
    # tag runs with no spaces.
    plain = _TAG_STRIP_RE.sub(" ", text or "")
    if not plain.strip():
        return False
    if _REPLACEMENT in plain:
        return True
    if _LONG_TOKEN_RE.search(plain):
        return True
    # Mojibake often shows up as dense high-byte runs in "latin" text.
    high = sum(1 for c in plain if ord(c) > 0x2500)
    return high > len(plain) * 0.05


_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<(t[dh])([^>]*)>(.*?)</t[dh]>", re.S | re.I)


def _span(attrs: str, name: str) -> int:
    m = re.search(name + r'=["\']?(\d+)', attrs, re.I)
    return max(1, int(m.group(1))) if m else 1


def table_grid(content: str) -> dict | None:
    """Parse an HTML table into rows with effective widths.

    A row's width counts colspan, plus columns still covered by a rowspan
    cell from a row above. Body rows (no <th>) are expected to agree on
    width; ``bad_rows`` are the body rows that don't match the most common
    width. Returns None for non-HTML tables (shape can't be judged).
    """
    raw_rows = _ROW_RE.findall(content or "")
    if not raw_rows:
        return None
    carry: list[list[int]] = []  # [rows_remaining, colspan] from rowspans above
    rows = []
    for raw in raw_rows:
        cells, width, new_carry = [], sum(c[1] for c in carry), []
        is_header = False
        for tag, attrs, inner in _CELL_RE.findall(raw):
            cs, rs = _span(attrs, "colspan"), _span(attrs, "rowspan")
            is_header = is_header or tag.lower() == "th"
            width += cs
            cells.append(" ".join(_TAG_STRIP_RE.sub(" ", inner).split()))
            if rs > 1:
                new_carry.append([rs - 1, cs])
        carry = [[n - 1, cs] for n, cs in carry if n > 1] + new_carry
        rows.append({"cells": cells, "width": width, "header": is_header})
    body = [i for i, r in enumerate(rows) if not r["header"]]
    if len(body) < 2:
        body = list(range(len(rows)))  # tiny table: compare everything
    widths = Counter(rows[i]["width"] for i in body)
    expected = widths.most_common(1)[0][0] if widths else 0
    return {"rows": rows, "expected_width": expected,
            "bad_rows": [i for i in body if rows[i]["width"] != expected]}


def _table_shape_ok(content: str) -> bool:
    """Body rows (non-header) must agree on effective column count.

    Header rows with merged cells legitimately differ, so only rows
    without <th> are compared; colspan and rowspan are both honored.
    """
    grid = table_grid(content)
    if grid is None:
        # Non-HTML serialization: can't judge shape, don't penalize.
        return True
    return not grid["bad_rows"] and grid["expected_width"] > 0


_EDGE_NUMBER_RE = re.compile(r"^\d{1,4}$")


def _leakage_count(blocks: list[dict], chunks: list[str]) -> int:
    """Chunks with a page number leaked into body text.

    Running headers ("Table of Contents") legitimately repeat words that
    also appear in body text, so substring matching is all false
    positives. Page numbers are different: a lone number on its own line
    at a chunk edge, matching a page_number block value, is never
    legitimate content.
    """
    page_nums = set()
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "page_number":
            t = _TAG_STRIP_RE.sub(" ", b.get("content") or "").strip()
            if _EDGE_NUMBER_RE.match(t):
                page_nums.add(t)
    if not page_nums:
        return 0
    leaked = 0
    for c in chunks:
        lines = _TAG_STRIP_RE.sub(" ", c).strip().splitlines()
        if not lines:
            continue
        edges = {lines[0].strip(), lines[-1].strip()}
        if edges & page_nums:
            leaked += 1
    return leaked


def fidelity_metrics(blocks: list[dict], chunks: list[str]) -> dict:
    """Intrinsic fidelity metrics for one parsed document."""
    pages: set[int] = set()
    table_blocks = 0
    table_shape_ok = 0
    ocr_confs: list[float] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        meta = b.get("metadata") or {}
        page = block_page_number(b)
        if page is not None:
            pages.add(page)
        conf = meta.get("avgOcrConfidence")
        if isinstance(conf, (int, float)):
            ocr_confs.append(float(conf))
        content = b.get("content") or ""
        if b.get("type") == "table":
            table_blocks += 1
            if _table_shape_ok(content):
                table_shape_ok += 1

    empty = sum(1 for c in chunks if len(c.strip()) < 20)
    garbled = sum(1 for c in chunks if _is_garbled(c))
    leaked = _leakage_count(blocks, chunks)

    n_chunks = len(chunks)
    return {
        "num_pages": len(pages),
        "num_blocks": len(blocks),
        "num_chunks": n_chunks,
        "empty_chunks": empty,
        "empty_rate": empty / n_chunks if n_chunks else 0.0,
        "garbled_chunks": garbled,
        "garbled_rate": garbled / n_chunks if n_chunks else 0.0,
        "table_blocks": table_blocks,
        "table_shape_ok": table_shape_ok,
        "table_shape_rate": table_shape_ok / table_blocks if table_blocks else 1.0,
        "ocr_confidence": (sum(ocr_confs) / len(ocr_confs)) if ocr_confs else None,
        "repeated_ngrams": leaked,
    }


_HYPHEN_BREAK_RE = re.compile(r"([a-z0-9]+)-[ \t]*\n\s*([a-z0-9]+)")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    """Markup stripped, Unicode folded, lowercased, thousands separators
    dropped, so "$1,382" and "1382" become the same word."""
    t = unicodedata.normalize("NFKC", _TAG_STRIP_RE.sub(" ", text or "")).lower()
    return _THOUSANDS_RE.sub("", t)


def text_coverage(source_text: str, parsed_text: str) -> float | None:
    """Fraction of the source's distinct words found in the parse output.

    Punctuation and currency are ignored. A word hyphenated across a line
    break counts as found whether the parse kept the hyphen ("COVID-19")
    or joined it ("development" from "devel-/opment").
    """
    src_text = _normalize(source_text)
    src = set(_WORD_RE.findall(src_text))
    if not src:
        return None
    parsed = set(_WORD_RE.findall(_normalize(parsed_text)))
    found = src & parsed
    for a, b in _HYPHEN_BREAK_RE.findall(src_text):
        if a + b in parsed:
            found |= {a, b}
    return len(found) / len(src)


# Thresholds behind each flag (tune against the corpus).
THRESHOLDS = {
    "table_shape_rate": 0.90,
    "empty_rate": 0.05,
    "ocr_confidence": 0.85,
    "text_coverage": 0.95,
}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def fidelity_checks(m: dict) -> list[dict]:
    """Every fidelity check as {key, label, value, threshold, ok, why}.

    ``ok`` is None when the check doesn't apply (no tables, no source PDF).
    red_flags() is the failing subset as one-line strings.
    """
    tables = m.get("table_blocks", 0) or 0
    tables_ok = m.get("table_shape_ok", 0) or 0
    shape = tables_ok / tables if tables else None
    empty, chunks = m.get("empty_chunks", 0) or 0, m.get("num_chunks", 0) or 0
    empty_rate = m.get("empty_rate")
    if empty_rate is None:
        empty_rate = empty / chunks if chunks else 0.0
    garbled = m.get("garbled_chunks", 0) or 0
    leaks = m.get("repeated_ngrams", 0) or 0
    ocr, cov = m.get("ocr_confidence"), m.get("text_coverage")
    return [
        {"key": "table_shape", "label": "Table shape",
         "value": (f"{tables_ok} of {_plural(tables, 'table')} consistent ({shape:.0%})"
                   if tables else "No tables found"),
         "threshold": f"≥ {THRESHOLDS['table_shape_rate']:.0%}",
         "ok": None if not tables else shape >= THRESHOLDS["table_shape_rate"],
         "why": "A table whose rows have different cell counts usually has merged or "
                "dropped cells, so figures can land in the wrong column."},
        {"key": "page_leaks", "label": "Page numbers in text",
         "value": _plural(leaks, "chunk"), "threshold": "0",
         "ok": leaks == 0,
         "why": "A page number at the start or end of a passage can be read as part "
                "of a figure or sentence."},
        {"key": "text_coverage", "label": "Text coverage",
         "value": "Not measured (source PDF not on server)" if cov is None else f"{cov:.1%} of source words",
         "threshold": f"≥ {THRESHOLDS['text_coverage']:.0%}",
         "ok": None if cov is None else cov >= THRESHOLDS["text_coverage"],
         "why": "Words in the PDF that the parse didn't capture can't be searched or cited."},
        {"key": "empty", "label": "Empty passages",
         "value": f"{empty} of {_plural(chunks, 'chunk')}",
         "threshold": f"≤ {THRESHOLDS['empty_rate']:.0%}",
         "ok": empty_rate <= THRESHOLDS["empty_rate"],
         "why": "Near-empty passages add noise to search without adding content."},
        {"key": "garbled", "label": "Garbled text",
         "value": _plural(garbled, "chunk"), "threshold": "0",
         "ok": garbled == 0,
         "why": "Broken characters usually mean an encoding or OCR failure."},
        {"key": "ocr", "label": "OCR confidence",
         "value": "Not a scan" if ocr is None else f"{ocr:.2f}",
         "threshold": f"≥ {THRESHOLDS['ocr_confidence']:.2f}",
         "ok": None if ocr is None else ocr >= THRESHOLDS["ocr_confidence"],
         "why": "Low OCR confidence means the page was read from an image and may "
                "contain misread characters."},
        {"key": "pages", "label": "Page numbers present",
         "value": "Yes" if (m.get("num_pages") or 0) else "No",
         "threshold": "Yes", "ok": bool(m.get("num_pages")),
         "why": "Without page numbers, answers can't point to where they came from."},
    ]


_FLAG_TEXT = {
    "table_shape": lambda m: (f"table shape inconsistent ({m['table_shape_ok']} of "
                              f"{_plural(m['table_blocks'], 'table')} ok)"),
    "page_leaks": lambda m: f"page numbers leaked into body text ({_plural(m['repeated_ngrams'], 'chunk')})",
    "text_coverage": lambda m: f"low text coverage vs source ({m['text_coverage']:.0%})",
    "empty": lambda m: f"high empty-chunk rate ({m.get('empty_rate') or 0:.0%})",
    "garbled": lambda m: f"garbled text ({_plural(m['garbled_chunks'], 'chunk')})",
    "ocr": lambda m: f"low OCR confidence ({m['ocr_confidence']:.2f}) — likely scan",
    "pages": lambda m: "no page numbers in parse output",
}


def red_flags(m: dict) -> list[str]:
    """Human-readable flags for a fidelity-metrics dict."""
    full = {"table_blocks": 0, "table_shape_ok": 0, "repeated_ngrams": 0,
            "garbled_chunks": 0, **m}
    return [_FLAG_TEXT[c["key"]](full) for c in fidelity_checks(full) if c["ok"] is False]


def fidelity_verdict(m: dict, pages: int, rate: float) -> str:
    """One plain sentence: what matters most and whether to re-parse."""
    failing = {c["key"] for c in fidelity_checks(m) if c["ok"] is False}
    cost = f"${pages * rate:.2f}"
    if not failing:
        return "Clean parse. No re-parse needed."
    if failing & {"table_shape", "ocr", "garbled"}:
        main = ("Tables are the main issue" if "table_shape" in failing
                else "Text quality is the main issue")
        return (f"{main}. The Performance engine handles tables and scans better, "
                f"so a re-parse is likely worth {cost}.")
    if failing == {"text_coverage"}:
        return ("Only coverage is low. Check the examples first: it can be "
                f"under-counted on slides and tables. Re-parse ({cost}) is optional.")
    return (f"Minor issues ({', '.join(sorted(failing)).replace('_', ' ')}). "
            f"Re-parse ({cost}) is optional.")


def check_golden_doc(spec: dict, chunks_text: str) -> dict:
    """Assert a golden document's probe facts and table shapes."""
    text = (chunks_text or "").lower()
    probes = []
    for probe in spec.get("probes", []):
        needles = probe.get("contains", [])
        found = [n for n in needles if n.lower() in text]
        probes.append({
            "fact": probe.get("fact", ""),
            "passed": len(found) == len(needles) and bool(needles),
            "missing": [n for n in needles if n.lower() not in text],
        })
    tables = []
    for t in spec.get("tables", []):
        tables.append({
            "page": t.get("page"),
            "description": t.get("description", ""),
            # Shape verification needs the golden doc's blocks; recorded
            # here as spec'd so the dashboard can show intent vs result.
            "expected_rows": t.get("rows"),
            "expected_cols": t.get("cols"),
        })
    passed = sum(1 for p in probes if p["passed"])
    return {
        "doc": spec.get("doc", ""),
        "doc_type": spec.get("doc_type", ""),
        "probes": probes,
        "probes_passed": passed,
        "probes_total": len(probes),
        "tables": tables,
        "ok": passed == len(probes) and len(probes) > 0,
    }


def load_golden_docs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
