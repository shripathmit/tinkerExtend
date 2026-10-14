"""Tests for the admin console access code."""
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


def test_admin_page_shows_login_until_signed_in():
    c = _client()
    r = c.get("/admin")
    assert r.status_code == 200
    assert 'action="/admin/login"' in r.text


def test_wrong_code_rejected():
    c = _client()
    r = c.post("/admin/login", data={"code": "nope"})
    assert r.status_code == 401
    assert c.get("/api/admin/jobs").status_code == 401


def test_login_sets_session_and_logout_clears_it():
    c = _client()
    r = c.post("/admin/login", data={"code": admin.ADMIN_CODE},
               follow_redirects=False)
    assert r.status_code == 303
    assert admin.ADMIN_CODE not in r.headers["set-cookie"]
    assert c.get("/api/admin/jobs").status_code == 200
    assert "tinkerExtend Console" in c.get("/admin").text
    assert 'action="/admin/login"' not in c.get("/admin").text
    c.get("/admin/logout")
    assert c.get("/api/admin/jobs").status_code == 401


def test_header_code_works_for_scripts():
    c = _client()
    r = c.get("/api/admin/jobs", headers={"X-Admin-Code": admin.ADMIN_CODE})
    assert r.status_code == 200
