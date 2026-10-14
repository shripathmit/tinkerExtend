"""Tests for the console access code (JSON login, cookie session, header)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinker_extend import admin


def _client():
    app = FastAPI()
    app.include_router(admin.router)
    return TestClient(app)


def test_admin_api_requires_code():
    c = _client()
    assert c.get("/api/admin/jobs").status_code == 401
    assert c.post("/api/admin/waves", json={"wave": "x"}).status_code == 401
    assert c.post("/api/admin/reparse", json={"doc_ids": ["x"]}).status_code == 401
    assert c.get("/api/admin/golden").status_code == 401


def test_session_reports_admin_state():
    c = _client()
    assert c.get("/api/session").json() == {"admin": False}
    c.post("/api/admin/login", json={"code": admin.ADMIN_CODE})
    assert c.get("/api/session").json() == {"admin": True}


def test_admin_page_redirects_into_app():
    r = _client().get("/admin", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/#overview"


def test_wrong_code_rejected():
    c = _client()
    r = c.post("/api/admin/login", json={"code": "nope"})
    assert r.status_code == 401
    assert r.json()["error"] == "Wrong code."
    assert c.get("/api/admin/jobs").status_code == 401


def test_login_sets_session_and_logout_clears_it():
    c = _client()
    r = c.post("/api/admin/login", json={"code": admin.ADMIN_CODE})
    assert r.status_code == 200
    assert admin.ADMIN_CODE not in r.headers["set-cookie"]
    assert c.get("/api/admin/jobs").status_code == 200
    c.post("/api/admin/logout")
    assert c.get("/api/admin/jobs").status_code == 401


def test_legacy_form_login_still_works():
    c = _client()
    r = c.post("/admin/login", data={"code": admin.ADMIN_CODE}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/#overview"
    assert c.get("/api/admin/jobs").status_code == 200


def test_header_code_works_for_scripts():
    c = _client()
    r = c.get("/api/admin/jobs", headers={"X-Admin-Code": admin.ADMIN_CODE})
    assert r.status_code == 200
