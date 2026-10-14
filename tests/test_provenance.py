"""Tests for the trust UI: safe HTML rendering, page mapping, feedback."""
import json
import sys

sys.path.insert(0, "src")

from tinker_extend.htmlfmt import to_safe_html
from tinker_extend.pages import page_boxes


def _blocks():
    return json.dumps([
        {"type": "heading", "content": "# Title",
         "boundingBox": {"left": 10, "top": 20, "right": 100, "bottom": 40},
         "metadata": {"page": {"number": 3, "width": 1240, "height": 1754}}},
        {"type": "table", "content": "<table><tr><td>$ 350</td></tr></table>",
         "boundingBox": {"left": 10, "top": 50, "right": 500, "bottom": 200},
         "metadata": {"page": {"number": 3, "width": 1240, "height": 1754}}},
        {"type": "text", "content": "later",
         "boundingBox": {"left": 10, "top": 10, "right": 100, "bottom": 30},
         "metadata": {"page": {"number": 7, "width": 1240, "height": 1754}}},
    ])


def test_page_boxes_groups_by_page():
    pages = page_boxes(_blocks())
    assert set(pages) == {3, 7}
    assert len(pages[3]["boxes"]) == 2
    assert pages[3]["width"] == 1240
    assert pages[7]["boxes"][0]["type"] == "text"


def test_page_boxes_bad_input():
    assert page_boxes("") == {}
    assert page_boxes("not json") == {}
    assert page_boxes(json.dumps([{"no": "meta"}])) == {}


def test_tables_render_as_tables_not_markup():
    out = to_safe_html("<table><tbody><tr><td>$ 350</td></tr></tbody></table>")
    assert "<table>" in out and "<td>$ 350</td>" in out
    assert "&lt;table&gt;" not in out


def test_table_attributes_survive():
    out = to_safe_html('<td rowspan="2">x</td>')
    assert '<td rowspan="2">x</td>' in out


def test_markdown_headers_become_headers():
    out = to_safe_html("### Other revenue\nsome text")
    assert "<h4>Other revenue</h4>" in out


def test_dangerous_tags_stay_escaped():
    out = to_safe_html('<script>alert(1)</script><img src=x onerror=y>')
    assert "<script>" not in out and "<img" not in out
    assert "&lt;script&gt;" in out


def test_feedback_roundtrip(tmp_path):
    from tinker_extend.store import KnowledgeBase
    kb = KnowledgeBase(str(tmp_path / "t.sqlite3"))
    fid = kb.add_feedback("q?", "extractive", "a", ["c1"], -1, "wrong number")
    rows = kb.list_feedback()
    assert len(rows) == 1 and rows[0]["id"] == fid
    assert rows[0]["rating"] == -1 and rows[0]["comment"] == "wrong number"
    kb.close()
