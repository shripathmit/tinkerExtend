"""Tests for the clearer fidelity checks: rowspan-aware table shape,
normalized coverage, structured checks, flag wording and verdict."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tinker_extend.fidelity import (
    _table_shape_ok, fidelity_checks, fidelity_verdict, red_flags, table_grid, text_coverage,
)

ROWSPAN_TABLE = """<table>
<tr><th>Program</th><th>Phase</th><th>Status</th></tr>
<tr><td rowspan="2">mRNA-1010</td><td>Ph 3</td><td>Filed</td></tr>
<tr><td>Ph 2</td><td>Ongoing</td></tr>
<tr><td>mRNA-1083</td><td>Approved</td><td>EU</td></tr>
</table>"""

RAGGED_TABLE = """<table>
<tr><td>a</td><td>b</td><td>c</td></tr>
<tr><td>d</td><td>e</td><td>f</td></tr>
<tr><td>g</td><td>h</td></tr>
<tr><td>i</td><td>j</td><td>k</td></tr>
</table>"""


def test_rowspan_rows_are_not_flagged():
    assert _table_shape_ok(ROWSPAN_TABLE)
    grid = table_grid(ROWSPAN_TABLE)
    assert [r["width"] for r in grid["rows"]] == [3, 3, 3, 3]
    assert grid["bad_rows"] == []


def test_ragged_table_reports_the_bad_row():
    grid = table_grid(RAGGED_TABLE)
    assert grid["expected_width"] == 3
    assert grid["bad_rows"] == [2]
    assert grid["rows"][2]["cells"] == ["g", "h"]
    assert not _table_shape_ok(RAGGED_TABLE)
    assert table_grid("plain text, not html") is None


def test_coverage_ignores_formatting_differences():
    source = "Total revenue was $1,382 million.\nCOVID-\n19 vaccine sales (Q1)."
    parsed = "<table><tr><td>Total revenue</td><td>1382</td></tr></table> was million. COVID-19 vaccine sales Q1"
    assert text_coverage(source, parsed) == 1.0
    # Old behaviour would have counted "$1,382", "million.", "COVID-", "(Q1)." as missing.
    assert text_coverage("a b c d", "a b c") == 0.75


def _metrics(**kw):
    m = {"num_pages": 25, "num_chunks": 25, "empty_chunks": 0, "garbled_chunks": 0,
         "table_blocks": 9, "table_shape_ok": 6, "repeated_ngrams": 1,
         "text_coverage": 0.806, "ocr_confidence": None}
    m.update(kw)
    return m


def test_checks_carry_value_threshold_and_reason():
    checks = {c["key"]: c for c in fidelity_checks(_metrics())}
    t = checks["table_shape"]
    assert t["ok"] is False and t["value"] == "6 of 9 tables consistent (67%)"
    assert t["threshold"] == "≥ 90%" and "column" in t["why"]
    assert checks["page_leaks"]["value"] == "1 chunk"
    assert checks["text_coverage"]["value"] == "80.6% of source words"
    assert checks["ocr"]["ok"] is None  # not a scan: not applicable
    no_tables = {c["key"]: c for c in fidelity_checks(_metrics(table_blocks=0, table_shape_ok=0))}
    assert no_tables["table_shape"]["ok"] is None


def test_flag_wording_and_plurals():
    flags = red_flags(_metrics())
    assert flags == ["table shape inconsistent (6 of 9 tables ok)",
                     "page numbers leaked into body text (1 chunk)",
                     "low text coverage vs source (81%)"]
    assert "(2 chunks)" in red_flags(_metrics(repeated_ngrams=2))[1]


def test_verdict_names_the_main_issue_and_cost():
    v = fidelity_verdict(_metrics(), pages=25, rate=0.025)
    assert v.startswith("Tables are the main issue")
    assert "$0.63" in v  # 25 pages x $0.025 = $0.625, rounded half-up like the UI
    assert "optional" in fidelity_verdict(
        _metrics(table_shape_ok=9, repeated_ngrams=0), pages=25, rate=0.025)
    assert fidelity_verdict(_metrics(table_shape_ok=9, repeated_ngrams=0, text_coverage=0.99),
                            pages=25, rate=0.025).startswith("Clean parse")
