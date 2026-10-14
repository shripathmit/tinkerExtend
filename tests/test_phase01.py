"""Tests for Phase 0/1: classifier, fidelity metrics, telemetry, waves."""
import json
import sys

sys.path.insert(0, "src")

from tinker_extend.classify import classify_content, classify_filename, mark_scan
from tinker_extend.fidelity import check_golden_doc, fidelity_metrics, red_flags, text_coverage
from tinker_extend.pipeline import DEFAULT_PROFILES, ENGINE_RATES
from tinker_extend.store import KnowledgeBase


def _blocks():
    return [
        {"type": "heading", "content": "# Title",
         "boundingBox": {"left": 10, "top": 20, "right": 100, "bottom": 40},
         "metadata": {"page": {"number": 1, "width": 1240, "height": 1754},
                      "avgOcrConfidence": 0.99}},
        {"type": "table",
         "content": "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>",
         "boundingBox": {"left": 10, "top": 50, "right": 500, "bottom": 200},
         "metadata": {"page": {"number": 1, "width": 1240, "height": 1754},
                      "avgOcrConfidence": 0.99}},
        {"type": "table",
         "content": "<table><tr><td>a</td></tr><tr><td>b</td><td>c</td></tr></table>",
         "boundingBox": {"left": 10, "top": 210, "right": 500, "bottom": 300},
         "metadata": {"page": {"number": 2, "width": 1240, "height": 1754}}},
    ]


def test_fidelity_metrics():
    m = fidelity_metrics(_blocks(), ["hello world, this is a chunk",
                                     "x" * 65 + "no-spaces-here"])
    assert m["num_pages"] == 2
    assert m["table_blocks"] == 2
    assert m["table_shape_ok"] == 1  # second table has ragged body rows
    assert m["table_shape_rate"] == 0.5
    assert m["garbled_chunks"] == 1  # 65-char token
    assert m["ocr_confidence"] == 0.99


def test_table_html_not_garbled():
    # Serialized tables have long tag runs with no spaces — not mojibake.
    # (Calibrated: longest legit token in the 10-Q after tag-stripping is 42 chars.)
    m = fidelity_metrics(
        [], ["<table><tbody><tr>" + "<td>cell</td>" * 30 + "</tr></tbody></table>"])
    assert m["garbled_chunks"] == 0
    # ...but a 200-char spaceless content run is still worth flagging.
    m2 = fidelity_metrics([], ["body " + "x" * 200])
    assert m2["garbled_chunks"] == 1


def test_merged_header_table_shape_ok():
    content = ('<table><tr><td rowspan="2"></td><th colspan="2">H</th></tr>'
               '<tr><th>2026</th><th>2025</th></tr>'
               '<tr><td>a</td><td>$ 1</td><td>$ 2</td></tr>'
               '<tr><td>b</td><td>$ 3</td><td>$ 4</td></tr></table>')
    m = fidelity_metrics(
        [{"type": "table", "content": content,
          "metadata": {"page": {"number": 1}}}], ["body text here"])
    assert m["table_shape_ok"] == 1


def test_red_flags():
    flags = red_flags({"empty_rate": 0.2, "garbled_rate": 0, "garbled_chunks": 0,
                       "table_shape_rate": 1.0, "num_pages": 3,
                       "ocr_confidence": 0.9, "repeated_ngrams": 0,
                       "text_coverage": 0.99})
    assert any("empty" in f for f in flags)
    assert not red_flags({"empty_rate": 0, "garbled_rate": 0, "garbled_chunks": 0,
                          "table_shape_rate": 1.0, "num_pages": 3,
                          "ocr_confidence": 0.99, "repeated_ngrams": 0,
                          "text_coverage": 0.99})


def test_text_coverage():
    assert text_coverage("a b c d", "a b c") == 0.75
    assert text_coverage("", "x") is None


def test_classify():
    t, conf, _ = classify_filename("Q1-2026-10-Q_final.pdf")
    assert t == "financial_filing" and conf >= 0.9
    t, _, _ = classify_content("This agreement is made whereas indemnification", "deal.pdf")
    assert t == "contract"
    t, _, _ = classify_filename("random_notes.txt")
    assert t == "other"
    assert mark_scan("report", 0.7) == "scan"
    assert mark_scan("report", 0.95) == "report"


def test_engine_profiles_cover_all_types():
    from tinker_extend.classify import DOC_TYPES
    for dt in DOC_TYPES:
        assert dt in DEFAULT_PROFILES, f"no engine profile for {dt}"
    for eng, rate in ENGINE_RATES.items():
        assert rate > 0


def test_telemetry_and_waves(tmp_path):
    from tinker_extend.models import Document
    kb = KnowledgeBase(str(tmp_path / "t.sqlite3"))
    kb.upsert_document(Document(doc_id="d1", source="/tmp/a.pdf", file_hash="d1"))
    kb.create_wave("pilot-100", 100)
    kb.record_telemetry("d1", "pilot-100", "financial_filing", "parse_light",
                        {"num_pages": 10, "num_chunks": 75, "table_blocks": 5,
                         "empty_chunks": 0, "garbled_chunks": 0,
                         "table_shape_ok": 5, "repeated_ngrams": 0,
                         "ocr_confidence": 0.99, "text_coverage": 0.98},
                        cost_usd=0.06, duration_s=12.0)
    rows = kb.telemetry()
    assert len(rows) == 1 and rows[0]["doc_type"] == "financial_filing"
    by_type = kb.fidelity_by_type()
    assert by_type[0]["docs"] == 1
    ov = kb.overview()
    assert ov["docs"] == 1 and ov["success_rate"] == 1.0
    kb.complete_wave("pilot-100")
    waves = kb.waves()
    assert waves[0]["status"] == "done" and waves[0]["ingested"] == 1
    kb.close()


def test_dead_letter_flow(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "t.sqlite3"))
    kb.add_dead_letter("/tmp/broken.pdf", "pilot-100", "ExtendRunFailed: boom")
    kb.add_dead_letter("/tmp/broken.pdf", "pilot-100", "ExtendRunFailed: boom again")
    rows = kb.list_dead_letters()
    assert len(rows) == 1 and rows[0]["attempts"] == 2
    requeued = kb.requeue_dead_letter(rows[0]["id"])
    assert requeued["source"] == "/tmp/broken.pdf"
    assert kb.list_dead_letters() == []
    kb.close()


def test_golden_doc_check():
    spec = {"doc": "x.pdf", "doc_type": "financial_filing",
            "probes": [{"fact": "R&D down", "contains": ["decreased by $207 million"]}],
            "tables": [{"page": 1, "description": "t", "rows": 2, "cols": 2}]}
    res = check_golden_doc(spec, "R&D expenses decreased by $207 million, or 24%")
    assert res["ok"] and res["probes_passed"] == 1
    res2 = check_golden_doc(spec, "nothing relevant here")
    assert not res2["ok"] and res2["probes"][0]["missing"] == ["decreased by $207 million"]


def test_pages_from_blocks_metadata_shape():
    # Regression: real Extend blocks nest the page under metadata.page.number;
    # the old code only looked for a flat top-level "page" key and found nothing.
    from tinker_extend.chunking import _pages_from_blocks, num_pages_in_run
    blocks = [
        {"type": "text", "content": "a",
         "metadata": {"page": {"number": 3, "width": 100, "height": 100}}},
        {"type": "text", "content": "b",
         "metadata": {"page": {"number": 5, "width": 100, "height": 100}}},
    ]
    assert _pages_from_blocks(blocks) == [3, 5]
    run = {"output": {"chunks": [{"content": "x", "blocks": blocks}]}}
    assert num_pages_in_run(run) == 2


def test_update_telemetry_metrics(tmp_path):
    from tinker_extend.models import Document
    kb = KnowledgeBase(str(tmp_path / "t.sqlite3"))
    kb.upsert_document(Document(doc_id="d9", source="/tmp/z.pdf", file_hash="d9"))
    kb.record_telemetry("d9", "w1", "other", "parse_light",
                        {"num_pages": 0, "num_chunks": 5}, cost_usd=0.0,
                        duration_s=3.0)
    ok = kb.update_telemetry_metrics("d9", {"num_pages": 12, "num_chunks": 5,
                                            "table_blocks": 2, "empty_chunks": 0,
                                            "garbled_chunks": 0, "text_coverage": 0.97,
                                            "ocr_confidence": 0.99,
                                            "table_shape_ok": 2,
                                            "repeated_ngrams": 0})
    assert ok
    row = kb.telemetry()[0]
    assert row["num_pages"] == 12 and row["text_coverage"] == 0.97
    assert row["wave"] == "w1" and row["duration_s"] == 3.0  # preserved
    kb.close()


def test_recompute_doc_telemetry(tmp_path):
    # Regression: recompute_doc_telemetry referenced fidelity_metrics without
    # importing it (NameError in production).
    import json as _json
    from tinker_extend.admin import recompute_doc_telemetry
    from tinker_extend.models import Chunk, Document
    kb = KnowledgeBase(str(tmp_path / "t.sqlite3"))
    kb.upsert_document(Document(doc_id="d10", source="/tmp/z.pdf", file_hash="d10"))
    blocks = [{"type": "text", "content": "hello world, this is a real chunk",
               "metadata": {"page": {"number": 2, "width": 10, "height": 10}}}]
    kb.conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, ordinal, content, blocks_json,"
        " pages_json, token_estimate) VALUES (?,?,?,?,?,?,?)",
        ("d10:00000", "d10", 0, "hello world, this is a real chunk",
         _json.dumps(blocks), "[2]", 8))
    kb.conn.commit()
    kb.record_telemetry("d10", "w1", "other", "parse_light",
                        {"num_pages": 0, "num_chunks": 1}, cost_usd=0.0,
                        duration_s=1.0)
    res = recompute_doc_telemetry(kb, "d10")
    assert res["metrics"]["num_pages"] == 1
    assert res["metrics"]["num_chunks"] == 1
    assert kb.telemetry()[0]["num_pages"] == 1
    assert recompute_doc_telemetry(kb, "missing") is None
    kb.close()
