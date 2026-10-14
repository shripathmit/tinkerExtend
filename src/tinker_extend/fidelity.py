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

from .chunking import block_page_number

_REPLACEMENT = "\ufffd"
_LONG_TOKEN_RE = re.compile(r"\S{60,}")
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


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


def _effective_cols(row_html: str) -> int:
    """Column count of a <tr>, honoring colspan."""
    total = 0
    for m in re.finditer(r"<t[dh]([^>]*)>", row_html, re.I):
        attrs = m.group(1)
        cs = re.search(r'colspan=["\']?(\d+)', attrs, re.I)
        total += int(cs.group(1)) if cs else 1
    return total


def _table_shape_ok(content: str) -> bool:
    """Body rows (non-header) must agree on effective column count.

    Header rows with merged cells (colspan/rowspan) legitimately differ,
    so only <tr>s without <th> are compared.
    """
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", content, re.S | re.I)
    if not rows:
        # Non-HTML serialization: can't judge shape, don't penalize.
        return True
    body = [r for r in rows if "<th" not in r.lower()]
    if len(body) < 2:
        body = rows  # tiny table: compare everything
    widths = {_effective_cols(r) for r in body}
    return len(widths) == 1 and next(iter(widths)) > 0


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


def text_coverage(source_text: str, parsed_text: str) -> float | None:
    """Fraction of source-text words recovered in the parse output."""
    src = set((source_text or "").lower().split())
    if not src:
        return None
    parsed = set((parsed_text or "").lower().split())
    return len(src & parsed) / len(src)


def red_flags(m: dict) -> list[str]:
    """Human-readable flags for a fidelity-metrics dict."""
    flags = []
    if m.get("empty_rate", 0) > 0.05:
        flags.append(f"high empty-chunk rate ({m['empty_rate']:.0%})")
    if m.get("garbled_rate", 0) > 0:
        flags.append(f"garbled chunks detected ({m['garbled_chunks']})")
    if m.get("table_shape_rate", 1) < 0.9:
        flags.append(f"table shape inconsistent ({m['table_shape_rate']:.0%} ok)")
    if m.get("num_pages", 0) == 0:
        flags.append("no page numbers in parse output")
    if (m.get("ocr_confidence") or 1.0) < 0.85:
        flags.append(f"low OCR confidence ({m['ocr_confidence']:.2f}) — likely scan")
    if m.get("repeated_ngrams", 0) > 0:
        flags.append(f"page numbers leaked into body text ({m['repeated_ngrams']} chunks)")
    cov = m.get("text_coverage")
    if cov is not None and cov < 0.95:
        flags.append(f"low text coverage vs source ({cov:.0%})")
    return flags


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
