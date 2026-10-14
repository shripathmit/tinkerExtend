"""Tests for the unified UI's API: page viewer, examples, synthesize flag,
feedback assertions, golden suite, and Performance re-parse."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import json
import time

import pytest
from fastapi.testclient import TestClient

from test_embeddings import FakeEmbedder, RD, SHIP
from tinker_extend import admin, embeddings, web
from tinker_extend.ingest import sha256_file
from tinker_extend.models import Chunk, Document
from tinker_extend.store import KnowledgeBase


def _blocks(page, text, kind="text", box=(10, 10, 200, 40)):
    return [{"type": kind, "content": text,
             "boundingBox": dict(zip(("left", "top", "right", "bottom"), box)),
             "metadata": {"page": {"number": page, "width": 1000, "height": 1400}}}]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A KB with one 2-page document whose source file is on disk."""
    src = tmp_path / "seed" / "filing.pdf"
    src.parent.mkdir()
    src.write_bytes(b"%PDF-fake filing bytes")
    doc_id = sha256_file(str(src))
    kb_path = str(tmp_path / "kb.sqlite3")
    emb = FakeEmbedder()
    kb = KnowledgeBase(kb_path)
    kb.upsert_document(Document(doc_id=doc_id, source=str(src), file_hash=doc_id, num_pages=2))
    chunks = []
    for i, (text, page) in enumerate(((RD, 1), (SHIP, 1), ("Page two legal text about governing law.", 2))):
        chunks.append(Chunk(chunk_id=f"{doc_id}:{i:05d}", doc_id=doc_id, ordinal=i, content=text,
                            blocks_json=json.dumps(_blocks(page, text)), pages_json=f"[{page}]",
                            embedding=embeddings.to_blob(emb.embed([text])[0]),
                            embedding_model=emb.model))
    kb.replace_chunks(doc_id, chunks)
    kb.record_telemetry(doc_id, "seed", "financial_filing", "parse_light",
                        {"num_pages": 2, "num_chunks": 3, "repeated_ngrams": 1}, 0.0, 0.0)
    kb.close()

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "q.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"question": "How much did research and development expenses decrease?",
         "relevant_contains": ["decreased by $207 million"], "answer_contains": ["207"], "answerable": True},
        {"question": "Zebra migration routes in antarctica?", "relevant_contains": [],
         "answer_contains": [], "answerable": False},
    ]))
    img_dir = tmp_path / "pages"
    (img_dir / doc_id).mkdir(parents=True)
    (img_dir / doc_id / "p1.png").write_bytes(b"png")

    for mod in (web, admin):
        monkeypatch.setattr(mod, "KB_PATH", kb_path)
        monkeypatch.setattr(mod, "SEED_DIR", str(src.parent))
    monkeypatch.setattr(web, "PAGE_IMG_DIR", str(img_dir))
    monkeypatch.setattr(admin, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(admin, "GOLDEN_QA", str(eval_dir / "q.jsonl"))
    monkeypatch.setattr(admin, "GOLDEN_DOCS", str(eval_dir / "none.json"))
    monkeypatch.setattr(embeddings, "_SHARED", emb)
    monkeypatch.delenv("TINKER_LLM_API_KEY", raising=False)
    c = TestClient(web.app)
    c.headers["X-Admin-Code"] = admin.ADMIN_CODE
    return {"client": c, "doc_id": doc_id, "kb_path": kb_path, "src": str(src)}


def test_index_serves_unified_app(env):
    r = env["client"].get("/")
    assert r.status_code == 200
    assert 'id="view-qa"' in r.text and "/static/app.js" in r.text
    assert env["client"].get("/static/app.css").status_code == 200


def test_page_endpoint_returns_chunks_boxes_and_image(env):
    c, doc_id = env["client"], env["doc_id"]
    d = c.get(f"/api/page/{doc_id}/1").json()
    assert d["num_pages"] == 2 and d["width"] == 1000
    assert d["image_url"] == f"/page_images/{doc_id}/p1.png"
    assert [ch["ordinal"] for ch in d["chunks"]] == [0, 1]
    assert d["chunks"][0]["boxes"][0]["right"] == 200
    assert "207 million" in d["chunks"][0]["content_html"]
    p2 = c.get(f"/api/page/{doc_id}/2").json()
    assert p2["image_url"] is None and len(p2["chunks"]) == 1
    assert c.get("/api/page/nope/1").status_code == 404


def test_examples_are_answerable_golden_questions(env):
    ex = env["client"].get("/api/examples").json()["examples"]
    assert ex == ["How much did research and development expenses decrease?"]


def test_stats_reports_llm_and_ask_respects_synthesize(env, monkeypatch):
    c = env["client"]
    assert c.get("/api/stats").json()["llm_configured"] is False

    class FakeLLM:
        def chat(self, system, user):
            return "Down $207M [1]."
    monkeypatch.setattr(web, "llm_from_env", lambda: FakeLLM())
    q = {"question": "How much did research and development expenses decrease?"}
    gen = c.post("/api/ask", json=q).json()
    assert gen["mode"] == "generated" and 'class="cite"' in gen["answer_html"]
    assert gen["search"] == "hybrid"
    ext = c.post("/api/ask", json={**q, "synthesize": False}).json()
    assert ext["mode"] == "extractive"
    assert ext["passages"][0]["doc_id"] == env["doc_id"]


def test_feedback_becomes_golden_assertions(env):
    c = env["client"]
    q = "How much did research and development expenses decrease?"
    target = f"{env['doc_id']}:00000"
    c.post("/api/feedback", json={"question": q, "rating": 1, "chunk_ids": [target],
                                  "active_chunk_id": target})
    bad = c.post("/api/feedback", json={"question": "shipping times", "rating": -1,
                                        "issue": "Wrong chunk", "comment": "cited refunds"}).json()
    g = c.get("/api/admin/golden").json()
    fb = [r for r in g["rows"] if r["source"] == "feedback"]
    assert {r["status"] for r in fb} == {"PASS", "FAIL"}
    fail = next(r for r in fb if r["status"] == "FAIL")
    assert fail["note"] == "Wrong chunk: cited refunds"
    qa = [r for r in g["rows"] if r["source"] == "golden_qa"]
    assert [r["status"] for r in qa] == ["PASS", "PASS"]  # answered + declined
    assert g["total"] == 4 and g["passed"] == 3
    assert g["compare"]["hybrid"] is not None

    # Resolving the thumbs-down drops it from the pass-rate denominator.
    assert c.post(f"/api/admin/feedback/{bad['id']}/resolve").status_code == 200
    g = c.get("/api/admin/golden").json()
    assert g["total"] == 3 and g["passed"] == 3
    assert any(r["status"] == "RESOLVED" for r in g["rows"])


def test_golden_run_is_stored_and_refreshed_when_chunks_change(env):
    c = env["client"]
    first = c.post("/api/admin/golden/run").json()
    assert c.get("/api/admin/golden").json()["ran_at"] == first["ran_at"]
    kb = KnowledgeBase(env["kb_path"])
    kb.set_meta("golden_last_run", json.dumps({**first, "ran_at": "stale"}))
    kb.replace_chunks(env["doc_id"], [])  # chunk write bumps vectors_rev
    kb.close()
    assert c.get("/api/admin/golden").json()["ran_at"] != "stale"


def test_doc_fidelity_lists_flags_and_cost(env):
    d = env["client"].get(f"/api/admin/doc/{env['doc_id']}/fidelity").json()
    assert d["name"] == "filing.pdf" and d["pages"] == 2
    assert d["estimated_cost_usd"] == 0.05
    assert d["source_available"] is True
    assert any("page numbers leaked" in f for f in d["flags"])
    assert d["history"] == []


class FakeExtend:
    engines: list = []

    def upload_file(self, path):
        return "file_1"

    def parse_async(self, file, engine=None, metadata=None, **kw):
        FakeExtend.engines.append(engine)
        return {"id": "run_2", "output": {"chunks": [
            {"content": RD + " (performance parse)", "blocks": _blocks(1, RD)}]}}


def test_reparse_with_performance_records_before_and_after(env, monkeypatch):
    FakeExtend.engines = []
    monkeypatch.setattr(admin, "_client_factory", lambda: FakeExtend())
    c, doc_id = env["client"], env["doc_id"]
    r = c.post("/api/admin/reparse", json={"doc_ids": [doc_id]}).json()
    assert r["ok"] and r["files"] == 1
    for _ in range(100):
        job = c.get("/api/admin/jobs").json()["jobs"][r["job"]]
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["progress"]["done"] == 1
    assert FakeExtend.engines == ["parse_performance"]
    d = c.get(f"/api/admin/doc/{doc_id}/fidelity").json()
    assert d["engine"] == "parse_performance"
    h = d["history"][0]
    assert h["before"]["engine"] == "parse_light" and h["before"]["num_chunks"] == 3
    assert h["after"]["num_chunks"] == 1 and h["engine"] == "parse_performance"
    page = c.get(f"/api/page/{doc_id}/1").json()
    assert "performance parse" in page["chunks"][0]["content_html"]


def test_reparse_refuses_when_source_missing_or_no_key(env, monkeypatch):
    c = env["client"]
    monkeypatch.setattr(admin, "_client_factory", lambda: FakeExtend())
    os.remove(env["src"])
    r = c.post("/api/admin/reparse", json={"doc_ids": [env["doc_id"]]})
    assert r.status_code == 400 and "source file not available" in r.json()["error"]

    def no_key():
        raise RuntimeError("EXTEND_API_KEY is not set.")
    monkeypatch.setattr(admin, "_client_factory", no_key)
    r = c.post("/api/admin/reparse", json={"doc_ids": [env["doc_id"]]})
    assert r.status_code == 400 and "EXTEND_API_KEY" in r.json()["error"]
