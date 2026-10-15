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


def test_compare_providers_dict_selection(monkeypatch, tmp_path):
    import tinker_extend.pipeline as pipe
    monkeypatch.setattr(pipe, "_source_text_words", lambda p: "alpha " * 100)

    seen = {}

    class Stub(providers.ParseProvider):
        name = "stub"

        def __init__(self, model=None):
            super().__init__(model)
            seen["model"] = self.model

        def parse(self, path):
            return ProviderResult(provider="stub", model=self.model,
                                  blocks=_blocks(), pages=2, cost_usd=0.01)

    monkeypatch.setitem(providers._PROVIDERS, "stub", Stub)
    out = compare_providers(str(tmp_path / "d.pdf"), {"stub": "stub-xl", "nope": "z"})
    assert set(out["providers"]) == {"stub"}
    assert seen["model"] == "stub-xl"
    assert out["providers"]["stub"]["model"] == "stub-xl"


def test_provider_status_has_models(monkeypatch):
    for k in ("EXTEND_API_KEY", "TINKER_OPENAI_API_KEY",
              "TINKER_GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    st = {p["name"]: p for p in provider_status()}
    assert st["openai"]["models"] == list(providers.OpenAIProvider.RATES)
    assert st["extend"]["models"] == ["parse_performance", "parse_light"]
    assert st["extend"]["default_model"] == "parse_performance"


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


# --- v0.9.3 fixes ---------------------------------------------------------

class _FakeHTTP:
    """Captures the request body and returns a canned JSON response."""

    def __init__(self, response: dict):
        self.response = response
        self.body = None

    def __call__(self, req, timeout=None):
        self.body = json.loads(req.data.decode())
        payload = json.dumps(self.response).encode()

        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def read(self_inner):
                return payload
        return _Resp()


def _pdf(tmp_path):
    # a real single-page PDF: parse() now counts pages via pypdf before
    # any HTTP call, so fake bytes no longer work here
    return _multipage_pdf(tmp_path, 1)


@pytest.mark.parametrize("cls, env, value", [
    (providers.OpenAIProvider, "TINKER_OPENAI_PARSE_MODEL", "gpt-5-mini"),
    (providers.GeminiProvider, "TINKER_GEMINI_PARSE_MODEL", "gemini-2.5-pro"),
    (providers.ExtendProvider, "TINKER_PLAYGROUND_EXTEND_ENGINE", "parse_light"),
])
def test_model_env_settings_are_honored(monkeypatch, cls, env, value):
    monkeypatch.delenv(env, raising=False)
    assert cls().model == cls.default_model
    monkeypatch.setenv(env, value)
    assert cls().model == value                    # env beats the class default
    assert cls(model=cls.default_model).model == cls.default_model  # dropdown beats env
    st = {p["name"]: p for p in provider_status()}
    assert st[cls.name]["default_model"] == value


@pytest.mark.parametrize("model, cap", [
    ("gpt-4o", 16_384), ("gpt-4o-mini", 16_384), ("gpt-5", 128_000), ("unknown-model", 16_384),
])
def test_openai_requests_stay_within_model_output_limit(monkeypatch, tmp_path, model, cap):
    monkeypatch.setenv("TINKER_OPENAI_API_KEY", "k")
    fake = _FakeHTTP({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
        {"blocks": [{"page": 1, "type": "text", "content": "hello"}]})}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    res = providers.OpenAIProvider(model=model).parse(_pdf(tmp_path))
    assert fake.body["max_completion_tokens"] == cap
    assert res.pages == 1 and res.blocks[0]["content"] == "hello"


@pytest.mark.parametrize("model, cap", [
    ("gemini-3.8-flash", 65_536), ("gemini-2.5-flash", 65_536), ("unknown-model", 8_192),
])
def test_gemini_requests_stay_within_model_output_limit(monkeypatch, tmp_path, model, cap):
    monkeypatch.setenv("TINKER_GEMINI_API_KEY", "k")
    fake = _FakeHTTP({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(
        {"blocks": [{"page": 1, "type": "text", "content": "hello"}]})}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}})
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    providers.GeminiProvider(model=model).parse(_pdf(tmp_path))
    assert fake.body["generationConfig"]["maxOutputTokens"] == cap


def test_truncated_llm_output_is_reported_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_OPENAI_API_KEY", "k")
    monkeypatch.setenv("TINKER_GEMINI_API_KEY", "k")
    monkeypatch.setattr(providers.urllib.request, "urlopen", _FakeHTTP(
        {"choices": [{"finish_reason": "length", "message": {"content": '{"blocks": [{"page": 1,'}}]}))
    with pytest.raises(RuntimeError, match=r"Output truncated: gpt-4o stopped at its 16,384-token"):
        providers.OpenAIProvider().parse(_pdf(tmp_path))
    monkeypatch.setattr(providers.urllib.request, "urlopen", _FakeHTTP(
        {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": '{"blocks": ['}]}}]}))
    with pytest.raises(RuntimeError, match=r"Output truncated: gemini-3.8-flash stopped at its 65,536-token"):
        providers.GeminiProvider().parse(_pdf(tmp_path))


def test_fewest_flags_ignores_checks_that_do_not_apply():
    base = {"ok": True, "fidelity": {"text_coverage": 0.9, "table_shape_rate": 1.0},
            "cost_usd": 0.05, "duration_s": 5.0}
    # "a" has two not-applicable checks (ok None) and no failures;
    # "b" has one real failure. "a" is cleaner.
    a = {**base, "checks": [{"ok": True}, {"ok": None}, {"ok": None}]}
    b = {**base, "checks": [{"ok": True}, {"ok": False}, {"ok": True}]}
    assert pick_winners({"a": a, "b": b})["fewest_flags"] == "a"


def test_extend_provider_uses_production_client_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("EXTEND_API_KEY", "ek")
    monkeypatch.setenv("EXTEND_WORKSPACE_ID", "ws_1")
    monkeypatch.setenv("EXTEND_API_VERSION", "2099-01-01")
    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)

        def upload_file(self, path):
            return "file_1"

        def parse_async(self, file, engine=None, metadata=None, **kw):
            seen["engine"] = engine
            return {"id": "run_1", "output": {"chunks": [{"blocks": [
                {"type": "text", "content": "x", "metadata": {"page": {"number": 1}}}]}]}}

    import tinker_extend.extend_client as ec
    monkeypatch.setattr(ec, "ExtendClient", FakeClient)
    res = providers.ExtendProvider(model="parse_light").parse(_pdf(tmp_path))
    assert seen["workspace_id"] == "ws_1" and seen["api_version"] == "2099-01-01"
    assert seen["engine"] == "parse_light"
    assert res.cost_usd == pytest.approx(0.00625)


def test_interrupted_provider_runs_are_failed_on_startup(tmp_path):
    from tinker_extend.store import KnowledgeBase
    kb = KnowledgeBase(str(tmp_path / "kb.sqlite3"))
    running = kb.create_provider_run("a.pdf", "/x/a.pdf", ["openai"])
    done = kb.create_provider_run("b.pdf", "/x/b.pdf", ["openai"])
    kb.finish_provider_run(done, {"providers": {}}, status="done")
    assert kb.fail_interrupted_provider_runs() == 1
    row = kb.get_provider_run(running)
    assert row["status"] == "error" and "restart" in row["result"]["error"]
    assert kb.get_provider_run(done)["status"] == "done"
    kb.close()


def test_best_tables_requires_tables():
    base = {"ok": True, "cost_usd": 0.05, "duration_s": 5.0, "checks": []}
    no_tables = {**base, "fidelity": {"text_coverage": 0.9, "table_shape_rate": 1.0, "table_blocks": 0}}
    some = {**base, "fidelity": {"text_coverage": 0.9, "table_shape_rate": 0.6, "table_blocks": 5}}
    assert pick_winners({"a": no_tables, "b": some})["table_shape"] == "b"
    assert pick_winners({"a": no_tables})["table_shape"] is None


def _multipage_pdf(tmp_path, n_pages):
    from pypdf import PdfWriter
    p = tmp_path / f"doc{n_pages}.pdf"
    w = PdfWriter()
    for _ in range(n_pages):
        w.add_blank_page(width=72, height=72)
    with open(p, "wb") as f:
        w.write(f)
    return str(p)


def test_extraction_prompt_names_page_range():
    assert "page by page" in providers.extraction_prompt()
    scoped = providers.extraction_prompt(11, 20)
    assert "pages 11 through 20" in scoped
    assert "ORIGINAL page numbers (11-20)" in scoped
    assert "never renumber from 1" in scoped


def test_chunk_ranges_single_for_small_pdf(tmp_path):
    path = _multipage_pdf(tmp_path, 5)
    assert providers._chunk_ranges(path, 10 * 1024 * 1024, 8) == [(0, 5)]


def test_chunk_ranges_split_by_pages_and_bytes(tmp_path):
    path = _multipage_pdf(tmp_path, 25)
    assert providers._chunk_ranges(path, 10 * 1024 * 1024, 10) == [
        (0, 10), (10, 20), (20, 25)]
    per_page = os.path.getsize(path) / 25
    ranges = providers._chunk_ranges(path, int(per_page * 2.5), 10)
    assert ranges[0] == (0, 2)
    assert ranges[-1][1] == 25
    assert sum(e - s for s, e in ranges) == 25


def test_globalize_block_pages_shifts_local_numbering():
    blocks = normalize_llm_blocks({"blocks": [
        {"page": 1, "type": "text", "content": "a"},
        {"page": 2, "type": "text", "content": "b"},
    ]})
    providers._globalize_block_pages(blocks, 11, 20)
    assert [b["metadata"]["page"]["number"] for b in blocks] == [11, 12]


def test_globalize_block_pages_keeps_global_numbering():
    blocks = normalize_llm_blocks({"blocks": [
        {"page": 11, "type": "text", "content": "a"},
        {"page": 12, "type": "text", "content": "b"},
    ]})
    providers._globalize_block_pages(blocks, 11, 20)
    assert [b["metadata"]["page"]["number"] for b in blocks] == [11, 12]


class _SequencedFakeHTTP:
    """Returns a different canned response per call; records request bodies."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data.decode()))
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        payload = json.dumps(resp).encode()

        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def read(self_inner):
                return payload
        return _Resp()


def _openai_ok(pages, in_tok=100, out_tok=50):
    blocks = [{"page": p, "type": "text", "content": f"page {p} text"}
              for p in pages]
    return {"choices": [{"finish_reason": "stop",
                         "message": {"content": json.dumps({"blocks": blocks})}}],
            "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok}}


def test_openai_splits_large_pdf_and_merges_chunks(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_OPENAI_API_KEY", "k")
    path = _multipage_pdf(tmp_path, 5)
    fake = _SequencedFakeHTTP([_openai_ok([1, 2]), _openai_ok([3, 4]),
                               _openai_ok([5])])
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    monkeypatch.setattr(providers.OpenAIProvider, "MAX_PAGES_PER_CHUNK", 2)
    res = providers.OpenAIProvider(model="gpt-5").parse(path)
    assert res.ok, res.error
    assert fake.calls == 3
    assert [b["metadata"]["page"]["number"] for b in res.blocks] == [1, 2, 3, 4, 5]
    assert res.input_tokens == 300 and res.output_tokens == 150
    assert res.raw_summary["chunks"] == 3
    texts = [b["messages"][0]["content"][0]["text"] for b in fake.bodies]
    assert "pages 1 through 2" in texts[0]
    assert "pages 3 through 4" in texts[1]
    assert "pages 5 through 5" in texts[2]


def test_openai_repairs_local_page_numbering(monkeypatch, tmp_path):
    # model ignores the global-numbering instruction; pages get shifted
    monkeypatch.setenv("TINKER_OPENAI_API_KEY", "k")
    path = _multipage_pdf(tmp_path, 4)
    fake = _SequencedFakeHTTP([_openai_ok([1, 2]), _openai_ok([1, 2])])
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    monkeypatch.setattr(providers.OpenAIProvider, "MAX_PAGES_PER_CHUNK", 2)
    res = providers.OpenAIProvider(model="gpt-5").parse(path)
    assert res.ok, res.error
    assert [b["metadata"]["page"]["number"] for b in res.blocks] == [1, 2, 3, 4]


def _gemini_ok(pages, in_tok=100, out_tok=50):
    blocks = [{"page": p, "type": "text", "content": f"page {p} text"}
              for p in pages]
    return {"candidates": [{"finishReason": "STOP",
                            "content": {"parts": [{"text": json.dumps({"blocks": blocks})}]}}],
            "usageMetadata": {"promptTokenCount": in_tok,
                              "candidatesTokenCount": out_tok}}


def test_gemini_splits_large_pdf_and_merges_chunks(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_GEMINI_API_KEY", "k")
    path = _multipage_pdf(tmp_path, 4)
    fake = _SequencedFakeHTTP([_gemini_ok([1, 2]), _gemini_ok([3, 4])])
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    monkeypatch.setattr(providers.GeminiProvider, "MAX_PAGES_PER_CHUNK", 2)
    res = providers.GeminiProvider().parse(path)
    assert res.ok, res.error
    assert fake.calls == 2
    assert [b["metadata"]["page"]["number"] for b in res.blocks] == [1, 2, 3, 4]
    assert res.input_tokens == 200 and res.output_tokens == 100
    assert res.raw_summary["chunks"] == 2


def test_gemini_reports_finish_reason_when_no_parts(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_GEMINI_API_KEY", "k")
    fake = _FakeHTTP({"candidates": [{"finishReason": "SAFETY"}]})
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    with pytest.raises(RuntimeError, match=r"finishReason=SAFETY"):
        providers.GeminiProvider().parse(_pdf(tmp_path))
