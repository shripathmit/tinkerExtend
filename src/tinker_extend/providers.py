"""Beta playground: parse-provider bake-off (the Data tab).

Production ingestion stays on Extend. This module lets an operator parse
the SAME document with several providers, normalizes every result to one
block shape, and scores each with the shared fidelity engine so the
comparison is apples-to-apples.

Providers:
  extend  - Extend.ai parse (the production path; engine via
            TINKER_PLAYGROUND_EXTEND_ENGINE, default parse_performance)
  openai  - multimodal LLM extraction via chat completions
            (TINKER_OPENAI_API_KEY, model TINKER_OPENAI_PARSE_MODEL)
  gemini  - multimodal LLM extraction via generateContent
            (TINKER_GEMINI_API_KEY, model TINKER_GEMINI_PARSE_MODEL)

Adding a provider: subclass ParseProvider, implement configured() and
parse(), register it in _PROVIDERS below.

Cost notes: Extend cost is exact (pages x engine rate). LLM costs are
approximate (tokens x list rates, overridable via env) and shown as such.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field

from .chunking import block_page_number
from .fidelity import fidelity_checks, fidelity_metrics, fidelity_verdict, text_coverage

PROVIDERS = ("extend", "openai", "gemini")

# One extraction prompt for every LLM provider, so the bake-off is fair.
# Tables come back as HTML so the shared table_shape check applies.
EXTRACTION_PROMPT = """You are a document parser. Extract the COMPLETE content of the attached PDF, page by page.
Return ONLY valid JSON (no markdown fences, no commentary):
{"blocks": [{"page": 1, "type": "text", "content": "..."}, {"page": 2, "type": "table", "content": "<table>...</table>"}]}
Rules:
- page is 1-based. One block per paragraph, heading, list, or table, in reading order.
- type is "text" or "table". Tables MUST be complete HTML <table> markup with <tr>, <th>, <td>, preserving headers.
- content is the verbatim text of the document. Do not summarize, shorten, or skip content.
- If a page has no extractable content, emit {"page": N, "type": "text", "content": ""}."""


def normalize_llm_blocks(payload: dict) -> list[dict]:
    """LLM JSON payload -> Extend-shaped blocks.

    {"type", "content", "metadata": {"page": {"number"}}} is the shape the
    fidelity engine understands.
    """
    blocks: list[dict] = []
    items = payload.get("blocks") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return blocks
    for b in items:
        if not isinstance(b, dict):
            continue
        page = b.get("page")
        blocks.append({
            "type": "table" if b.get("type") == "table" else "text",
            "content": str(b.get("content") or ""),
            "metadata": {"page": {"number": page if isinstance(page, int) else None}},
        })
    return blocks


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rsplit("```", 1)[0].strip():
            t = t.rsplit("```", 1)[0]
    return t.strip()


def _usd(in_tokens: int, out_tokens: int, rate_in: float, rate_out: float) -> float:
    return in_tokens / 1_000_000 * rate_in + out_tokens / 1_000_000 * rate_out


@dataclass
class ProviderResult:
    provider: str
    model: str = ""
    ok: bool = True
    error: str = ""
    blocks: list[dict] = field(default_factory=list)
    pages: int = 0
    cost_usd: float = 0.0
    cost_note: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    raw_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "ok": self.ok,
            "error": self.error,
            "num_blocks": len(self.blocks),
            "pages": self.pages,
            "cost_usd": round(self.cost_usd, 4),
            "cost_note": self.cost_note,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_s": round(self.duration_s, 1),
            "raw_summary": self.raw_summary,
        }


class ParseProvider:
    """Interface every bake-off provider implements."""

    name = "base"

    @classmethod
    def configured(cls) -> tuple[bool, str]:
        """(is_configured, reason_when_not). Never raises."""
        return False, "not implemented"

    @property
    def model(self) -> str:
        return ""

    def parse(self, path: str) -> ProviderResult:
        raise NotImplementedError


class ExtendProvider(ParseProvider):
    """Extend.ai parse — the production path, for reference in the bake-off."""

    name = "extend"

    def __init__(self) -> None:
        self.engine = os.environ.get("TINKER_PLAYGROUND_EXTEND_ENGINE", "parse_performance")

    @classmethod
    def configured(cls) -> tuple[bool, str]:
        key = os.environ.get("EXTEND_API_KEY", "")
        return (True, "") if key else (False, "Set EXTEND_API_KEY to enable")

    @property
    def model(self) -> str:
        return self.engine

    def parse(self, path: str) -> ProviderResult:
        from .extend_client import ExtendClient
        from .pipeline import ENGINE_RATES

        api_key = os.environ.get("EXTEND_API_KEY", "")
        if not api_key:
            raise RuntimeError("EXTEND_API_KEY not set")
        client = ExtendClient(api_key=api_key)
        file_id = client.upload_file(path)
        run = client.parse_async(
            {"id": file_id}, engine=self.engine,
            metadata={"source": os.path.basename(path), "playground": "1"})
        blocks: list[dict] = []
        for ch in (run.get("output") or {}).get("chunks") or []:
            blocks.extend(ch.get("blocks") or [])
        pages = len({p for p in (block_page_number(b) for b in blocks) if p is not None})
        return ProviderResult(
            provider=self.name, model=self.engine, blocks=blocks, pages=pages,
            cost_usd=pages * ENGINE_RATES.get(self.engine, 0.0),
            cost_note="exact (pages x engine rate)",
            raw_summary={"run_id": run.get("id"), "engine": self.engine})


class OpenAIProvider(ParseProvider):
    """Multimodal LLM extraction via OpenAI chat completions."""

    name = "openai"
    # Approximate list rates ($ per 1M tokens); override via env.
    RATES = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60),
             "gpt-5": (1.25, 10.00), "gpt-5-mini": (0.25, 2.00)}

    def __init__(self) -> None:
        self.api_key = os.environ.get("TINKER_OPENAI_API_KEY", "")
        self._model = os.environ.get("TINKER_OPENAI_PARSE_MODEL", "gpt-4o")

    @classmethod
    def configured(cls) -> tuple[bool, str]:
        key = os.environ.get("TINKER_OPENAI_API_KEY", "")
        return (True, "") if key else (False, "Set TINKER_OPENAI_API_KEY to enable")

    @property
    def model(self) -> str:
        return self._model

    def parse(self, path: str) -> ProviderResult:
        if not self.api_key:
            raise RuntimeError("TINKER_OPENAI_API_KEY not set")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        body = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "file", "file": {
                        "filename": os.path.basename(path),
                        "file_data": f"data:application/pdf;base64,{b64}"}},
                ],
            }],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 32000,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read().decode())
            except Exception:
                err = {}
            raise RuntimeError(
                f"OpenAI HTTP {exc.code}: {err.get('error', {}).get('message', exc.reason)}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
            payload = json.loads(_strip_fences(content))
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI returned unparseable output: {exc}") from exc
        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)
        rate_in = float(os.environ.get("TINKER_OPENAI_RATE_IN",
                                       self.RATES.get(self._model, (2.50, 10.00))[0]))
        rate_out = float(os.environ.get("TINKER_OPENAI_RATE_OUT",
                                        self.RATES.get(self._model, (2.50, 10.00))[1]))
        blocks = normalize_llm_blocks(payload)
        pages = len({p for p in (block_page_number(b) for b in blocks) if p is not None})
        return ProviderResult(
            provider=self.name, model=self._model, blocks=blocks, pages=pages,
            cost_usd=_usd(in_tok, out_tok, rate_in, rate_out),
            cost_note="approximate (tokens x list rate)",
            input_tokens=in_tok, output_tokens=out_tok,
            raw_summary={"model": self._model})


class GeminiProvider(ParseProvider):
    """Multimodal LLM extraction via Gemini generateContent."""

    name = "gemini"
    # Approximate list rates ($ per 1M tokens); override via env.
    RATES = {"gemini-2.0-flash": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50),
             "gemini-2.5-pro": (1.25, 10.00)}

    def __init__(self) -> None:
        self.api_key = (os.environ.get("TINKER_GEMINI_API_KEY", "")
                        or os.environ.get("GOOGLE_API_KEY", ""))
        self._model = os.environ.get("TINKER_GEMINI_PARSE_MODEL", "gemini-2.0-flash")

    @classmethod
    def configured(cls) -> tuple[bool, str]:
        key = os.environ.get("TINKER_GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        return (True, "") if key else (False, "Set TINKER_GEMINI_API_KEY to enable")

    @property
    def model(self) -> str:
        return self._model

    def parse(self, path: str) -> ProviderResult:
        if not self.api_key:
            raise RuntimeError("TINKER_GEMINI_API_KEY not set")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        body = {
            "contents": [{"parts": [
                {"text": EXTRACTION_PROMPT},
                {"inlineData": {"mimeType": "application/pdf", "data": b64}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "maxOutputTokens": 65536},
        }
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent",
            data=json.dumps(body).encode(),
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read().decode())
            except Exception:
                err = {}
            msg = (err.get("error") or {}).get("message", f"HTTP {exc.code}")
            raise RuntimeError(f"Gemini {msg}") from exc
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            payload = json.loads(_strip_fences(text))
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Gemini returned unparseable output: {exc}") from exc
        usage = data.get("usageMetadata") or {}
        in_tok = int(usage.get("promptTokenCount") or 0)
        out_tok = int(usage.get("candidatesTokenCount") or 0)
        rate_in = float(os.environ.get("TINKER_GEMINI_RATE_IN",
                                       self.RATES.get(self._model, (0.10, 0.40))[0]))
        rate_out = float(os.environ.get("TINKER_GEMINI_RATE_OUT",
                                        self.RATES.get(self._model, (0.10, 0.40))[1]))
        blocks = normalize_llm_blocks(payload)
        pages = len({p for p in (block_page_number(b) for b in blocks) if p is not None})
        return ProviderResult(
            provider=self.name, model=self._model, blocks=blocks, pages=pages,
            cost_usd=_usd(in_tok, out_tok, rate_in, rate_out),
            cost_note="approximate (tokens x list rate)",
            input_tokens=in_tok, output_tokens=out_tok,
            raw_summary={"model": self._model})


_PROVIDERS: dict[str, type[ParseProvider]] = {
    "extend": ExtendProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def provider_status() -> list[dict]:
    """For the Data tab: which providers can run right now."""
    out = []
    for name in PROVIDERS:
        cls = _PROVIDERS[name]
        ok, reason = cls.configured()
        try:
            model = cls().model
        except Exception:
            model = ""
        out.append({"name": name, "configured": ok, "reason": reason, "model": model})
    return out


def score_result(path: str, res: ProviderResult) -> dict:
    """Attach fidelity metrics/checks/verdict to a provider result.

    Blocks are chunked per page (in order), mirroring the production
    telemetry path: fidelity_metrics + source-text coverage.
    """
    from .pipeline import _source_text_words  # local import: pipeline never imports providers

    d = res.to_dict()
    d["blocks"] = res.blocks
    if not res.ok or not res.blocks:
        d.update({"fidelity": None, "checks": [],
                  "verdict": res.error or "no blocks produced"})
        return d
    per_page: dict[int, list[str]] = {}
    for b in res.blocks:
        pg = block_page_number(b)
        per_page.setdefault(pg if pg is not None else 0, []).append(b.get("content") or "")
    chunk_texts = ["\n".join(t for t in texts).strip()
                   for _, texts in sorted(per_page.items())]
    chunk_texts = [c for c in chunk_texts if c] or [""]
    m = fidelity_metrics(res.blocks, chunk_texts)
    src = _source_text_words(path)
    if src:
        m["text_coverage"] = text_coverage(src, "\n".join(chunk_texts))
    d["fidelity"] = m
    d["checks"] = fidelity_checks(m)
    pages = m.get("num_pages") or 0
    d["verdict"] = fidelity_verdict(m, pages, (res.cost_usd / pages) if pages else 0.0)
    return d


def pick_winners(scored: dict) -> dict:
    """Best provider per dimension, over successful runs only."""
    ok = {n: s for n, s in scored.items() if s.get("ok") and s.get("fidelity")}
    if not ok:
        return {}
    cov = lambda n: ok[n]["fidelity"].get("text_coverage")
    return {
        "coverage": max(ok, key=lambda n: cov(n) if cov(n) is not None else -1),
        "table_shape": max(ok, key=lambda n: ok[n]["fidelity"].get("table_shape_rate") or -1),
        "cheapest": min(ok, key=lambda n: ok[n]["cost_usd"]),
        "fastest": min(ok, key=lambda n: ok[n]["duration_s"]),
        "fewest_flags": min(ok, key=lambda n: sum(1 for c in ok[n]["checks"] if not c.get("ok"))),
    }


def compare_providers(path: str, names: list[str] | None = None) -> dict:
    """Parse `path` with each named provider and score them all."""
    import datetime

    names = [n for n in (names or list(PROVIDERS)) if n in _PROVIDERS] or list(PROVIDERS)
    scored: dict[str, dict] = {}
    for name in names:
        provider = _PROVIDERS[name]()
        started = time.monotonic()
        try:
            res = provider.parse(path)
        except Exception as exc:  # noqa: BLE001 - one provider must not sink the run
            res = ProviderResult(provider=name, model=getattr(provider, "model", ""),
                                 ok=False, error=f"{type(exc).__name__}: {exc}")
        res.duration_s = time.monotonic() - started
        scored[name] = score_result(path, res)
    return {
        "filename": os.path.basename(path),
        "ran_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "providers": scored,
        "winners": pick_winners(scored),
        "note": ("Beta playground. LLM costs are approximate (tokens x list rates). "
                 "Production ingestion continues on Extend."),
    }
