"""Tests for the beta parse-provider playground (Data tab)."""
import json
import os

import pytest

from tinker_extend import providers
from tinker_extend.providers import (
    ProviderResult,
    compare_providers,
    normalize_llm_blocks,
    pick_winners,
    provider_status,
    score_result,
)


def _blocks():
    return normalize_llm_blocks({"blocks": [
        {"page": 1, "type": "text",
         "content": "alpha beta gamma " * 30},
        {"page": 1, "type": "table",
         "content": "<table><tr><th>a</th><th>b</th></tr>"
                    "<tr><td>1</td><td>2</td></tr></table>"},
        {"page": 2, "type": "text", "content": "second page here " * 20},
    ]})


def test_normalize_llm_blocks():
    blocks = _blocks()
    assert len(blocks) == 3
    assert blocks[0]["metadata"]["page"]["number"] == 1
    assert blocks[1]["type"] == "table"
    # unknown types fall back to text; bad pages become None
    weird = normalize_llm_blocks({"blocks": [
        {"page": "x", "type": "figure", "content": "hi"},
        {"nope": True},
        "junk",
    ]})
    assert weird[0]["type"] == "text"
    assert weird[0]["metadata"]["page"]["number"] is None
    assert weird[1]["content"] == ""
    assert normalize_llm_blocks({}) == []
    assert normalize_llm_blocks({"blocks": "nope"}) == []


def test_strip_fences():
    assert providers._strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert providers._strip_fences('{"a": 1}') == '{"a": 1}'


def test_score_result_fidelity(monkeypatch):
    # score_result lazily imports _source_text_words from pipeline
    import tinker_extend.pipeline as pipe
    monkeypatch.setattr(pipe, "_source_text_words",
                        lambda p: "alpha beta gamma " * 100)
    res = ProviderResult(provider="openai", model="gpt-4o", blocks=_blocks(),
                         pages=2, cost_usd=0.05,
                         input_tokens=1000, output_tokens=500)
    s = score_result("/tmp/x.pdf", res)
    f = s["fidelity"]
    assert f["num_pages"] == 2
    assert f["table_blocks"] == 1
    assert f["text_coverage"] is not None
    assert isinstance(s["verdict"], str) and s["verdict"]
    assert isinstance(s["checks"], list) and s["checks"]
    assert s["num_blocks"] == 3


def test_score_result_failure():
    res = ProviderResult(provider="gemini", ok=False, error="boom")
    s = score_result("/tmp/x.pdf", res)
    assert s["fidelity"] is None
    assert s["verdict"] == "boom"
    assert s["ok"] is False


def test_pick_winners():
    good = {"ok": True,
            "fidelity": {"text_coverage": 0.9, "table_shape_rate": 0.8},
            "cost_usd": 0.05, "duration_s": 10.0, "checks": [{"ok": True}]}
    bad = {"ok": False, "fidelity": None, "cost_usd": 0.01,
           "duration_s": 1.0, "checks": []}
    w = pick_winners({"openai": good, "gemini": bad})
    assert w["coverage"] == "openai"
    assert w["cheapest"] == "openai"  # failed runs don't count
    assert pick_winners({"gemini": bad}) == {}


def test_provider_status_unconfigured(monkeypatch):
    for k in ("EXTEND_API_KEY", "TINKER_OPENAI_API_KEY",
              "TINKER_GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    st = {p["name"]: p for p in provider_status()}
    assert set(st) == {"extend", "openai", "gemini"}
    assert all(not p["configured"] for p in st.values())
    assert all(p["reason"] for p in st.values())


def test_compare_providers_stub(monkeypatch, tmp_path):
    import tinker_extend.pipeline as pipe
    monkeypatch.setattr(pipe, "_source_text_words", lambda p: "alpha " * 100)

    class Stub(providers.ParseProvider):
        name = "stub"

        @property
        def model(self):
            return "stub-1"

        def parse(self, path):
            return ProviderResult(provider="stub", model="stub-1",
                                  blocks=_blocks(), pages=2, cost_usd=0.01)

    class Boom(providers.ParseProvider):
        name = "boom"

        def parse(self, path):
            raise RuntimeError("kaput")

    monkeypatch.setitem(providers._PROVIDERS, "stub", Stub)
    monkeypatch.setitem(providers._PROVIDERS, "boom", Boom)
    out = compare_providers(str(tmp_path / "d.pdf"), ["stub", "boom", "nope"])
    assert set(out["providers"]) == {"stub", "boom"}  # unknown names ignored
    assert out["providers"]["stub"]["ok"] is True
    assert out["providers"]["stub"]["fidelity"]["num_pages"] == 2
    assert out["providers"]["boom"]["ok"] is False
    assert "kaput" in out["providers"]["boom"]["error"]
    assert out["winners"]["coverage"] == "stub"
    assert out["filename"] == "d.pdf"


def test_provider_run_store(tmp_path):
    from tinker_extend.store import KnowledgeBase
    kb = KnowledgeBase(str(tmp_path / "kb.sqlite3"))
    try:
        rid = kb.create_provider_run("d.pdf", "/tmp/d.pdf", ["extend", "openai"])
        runs = kb.list_provider_runs()
        assert len(runs) == 1 and runs[0]["status"] == "running"
        assert runs[0]["providers"] == ["extend", "openai"]
        assert "result_json" not in runs[0]  # list stays light
        kb.finish_provider_run(rid, {"winners": {"coverage": "extend"}})
        row = kb.get_provider_run(rid)
        assert row["status"] == "done"
        assert row["result"]["winners"]["coverage"] == "extend"
        assert kb.get_provider_run(9999) is None
    finally:
        kb.close()
