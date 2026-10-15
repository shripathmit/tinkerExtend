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
import urllib.error
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


def extraction_prompt(page_start: int | None = None,
                      page_end: int | None = None) -> str:
    """The shared extraction prompt, optionally scoped to a page range.

    Chunk prompts name the ORIGINAL 1-based page numbers so blocks merged
    from several requests keep global numbering.
    """
    if page_start is None:
        return EXTRACTION_PROMPT
    return (
        f"The attached PDF is an excerpt: pages {page_start} through {page_end} "
        f"of the original document. In the \"page\" field of every block, use the "
        f"ORIGINAL page numbers ({page_start}-{page_end}); never renumber from 1.\n\n"
        + EXTRACTION_PROMPT
    )


def _pdf_page_count(path: str) -> int:
    from pypdf import PdfReader
    try:
        return len(PdfReader(path).pages)
    except Exception as exc:
        raise RuntimeError(f"could not read PDF page count: {exc}") from exc


def _write_page_range(src: str, start: int, end: int, dest: str) -> None:
    """Copy pages [start, end) (0-based) of src into a new PDF at dest."""
    from pypdf import PdfReader, PdfWriter
    try:
        reader = PdfReader(src)
        writer = PdfWriter()
        for p in reader.pages[start:end]:
            writer.add_page(p)
        with open(dest, "wb") as f:
            writer.write(f)
    except Exception as exc:
        raise RuntimeError(
            f"could not split PDF pages {start + 1}-{end}: {exc}") from exc


def _chunk_ranges(path: str, max_bytes_per_chunk: int,
                  max_pages_per_chunk: int) -> list[tuple[int, int]]:
    """Split a PDF into (start, end) 0-based page ranges.

    Each chunk stays under max_bytes_per_chunk (estimated from the average
    bytes/page) and holds at most max_pages_per_chunk pages, so no single
    LLM request exceeds the model's input or output token limits.
    """
    total_pages = _pdf_page_count(path)
    if total_pages <= 0:
        raise RuntimeError("PDF has no readable pages")
    bytes_per_page = max(os.path.getsize(path), 1) / total_pages
    pages_by_bytes = max(1, int(max_bytes_per_chunk / bytes_per_page))
    per_chunk = max(1, min(pages_by_bytes, max_pages_per_chunk))
    return [(s, min(s + per_chunk, total_pages))
            for s in range(0, total_pages, per_chunk)]


def _globalize_block_pages(blocks: list[dict], start_1: int, end_1: int) -> None:
    """Shift chunk blocks that look 1-based-local to global page numbers.

    The chunk prompt asks for original page numbers, but if every block page
    falls inside 1..chunk_span instead of start_1..end_1, assume the model
    renumbered locally and shift. Mutates blocks in place; a no-op for the
    first chunk (start_1 == 1).
    """
    span = end_1 - start_1 + 1
    pages = [p for p in (block_page_number(b) for b in blocks) if p is not None]
    if not pages:
        return
    if max(pages) <= span and (min(pages) < start_1 or max(pages) > end_1):
        for b in blocks:
            pg = (b.get("metadata") or {}).get("page") or {}
            if isinstance(pg.get("number"), int):
                pg["number"] += start_1 - 1


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


def truncation_message(model: str, limit: int) -> str:
    """Error shown when an LLM stops at its output limit mid-document."""
    return (f"Output truncated: {model} stopped at its {limit:,}-token output limit before "
            "finishing the document. Try a shorter PDF or a model with a larger output limit.")


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
    """Interface every bake-off provider implements.

    `model` selects the extraction model/engine for this run; when None
    the provider default is used. The Data tab offers available_models()
    in a dropdown.
    """

    name = "base"
    default_model = ""
    model_env = ""  # env var that overrides default_model, e.g. TINKER_OPENAI_PARSE_MODEL

    def __init__(self, model: str | None = None) -> None:
        # Explicit choice (Data tab dropdown) > env setting > class default.
        self._model = model or self.env_default()

    @classmethod
    def env_default(cls) -> str:
        return (os.environ.get(cls.model_env, "") if cls.model_env else "") or cls.default_model

    @classmethod
    def configured(cls) -> tuple[bool, str]:
        """(is_configured, reason_when_not). Never raises."""
        return False, "not implemented"

    @classmethod
    def available_models(cls) -> list[str]:
        """Model ids the UI offers for this provider."""
        return [cls.default_model] if cls.default_model else []

    @property
    def model(self) -> str:
        return self._model

    def parse(self, path: str) -> ProviderResult:
        raise NotImplementedError


class ExtendProvider(ParseProvider):
    """Extend.ai parse — the production path, for reference in the bake-off."""

    name = "extend"
    default_model = "parse_performance"
    model_env = "TINKER_PLAYGROUND_EXTEND_ENGINE"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model)
        self.engine = self._model

    @classmethod
    def configured(cls) -> tuple[bool, str]:
        key = os.environ.get("EXTEND_API_KEY", "")
        return (True, "") if key else (False, "Set EXTEND_API_KEY to enable")

    @classmethod
    def available_models(cls) -> list[str]:
        return ["parse_performance", "parse_light"]

    @property
    def model(self) -> str:
        return self.engine

    def parse(self, path: str) -> ProviderResult:
        from .config import load_config
        from .extend_client import ExtendClient
        from .pipeline import ENGINE_RATES

        cfg = load_config()  # raises with a clear message if EXTEND_API_KEY is unset
        # Same client settings as production ingestion (API version,
        # workspace, base URL), so workspace-scoped keys work here too.
        client = ExtendClient(api_key=cfg.api_key, api_version=cfg.api_version,
                              base_url=cfg.base_url, workspace_id=cfg.workspace_id)
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

    # Maximum output tokens per model; requests above the limit are rejected.
    MAX_OUTPUT = {"gpt-4o": 16_384, "gpt-4o-mini": 16_384,
                  "gpt-5": 128_000, "gpt-5-mini": 128_000}
    DEFAULT_MAX_OUTPUT = 16_384

    # Input token limits per model (gpt-5 reports 272k; 4o-class is 128k).
    # Large PDFs are split into page-range chunks so no request exceeds
    # CHUNK_FRACTION of the input limit (base64 inflates bytes ~4/3 and
    # tokenizes roughly 1 token per 3.5 chars -> ~0.4 tokens/byte).
    INPUT_LIMIT = {"gpt-4o": 128_000, "gpt-4o-mini": 128_000,
                   "gpt-5": 272_000, "gpt-5-mini": 272_000}
    DEFAULT_INPUT_LIMIT = 128_000
    CHUNK_FRACTION = 0.70
    TOKENS_PER_BYTE = 0.4
    MAX_PAGES_PER_CHUNK = 8

    default_model = "gpt-4o"
    model_env = "TINKER_OPENAI_PARSE_MODEL"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model)
        self.api_key = os.environ.get("TINKER_OPENAI_API_KEY", "")

    def max_output_tokens(self) -> int:
        return self.MAX_OUTPUT.get(self._model, self.DEFAULT_MAX_OUTPUT)

    def max_input_tokens(self) -> int:
        return self.INPUT_LIMIT.get(self._model, self.DEFAULT_INPUT_LIMIT)

    def _chunk_byte_budget(self) -> int:
        return int(self.max_input_tokens() * self.CHUNK_FRACTION / self.TOKENS_PER_BYTE)

    @classmethod
    def configured(cls) -> tuple[bool, str]:
        key = os.environ.get("TINKER_OPENAI_API_KEY", "")
        return (True, "") if key else (False, "Set TINKER_OPENAI_API_KEY to enable")

    @classmethod
    def available_models(cls) -> list[str]:
        return list(cls.RATES)

    def parse(self, path: str) -> ProviderResult:
        if not self.api_key:
            raise RuntimeError("TINKER_OPENAI_API_KEY not set")
        ranges = _chunk_ranges(path, self._chunk_byte_budget(),
                               self.MAX_PAGES_PER_CHUNK)
        blocks: list[dict] = []
        in_tok = out_tok = 0
        if len(ranges) == 1:
            b, i, o = self._parse_one(path, extraction_prompt())
            blocks, in_tok, out_tok = b, i, o
        else:
            import tempfile
            with tempfile.TemporaryDirectory(prefix="tinker-playground-") as tmp:
                for n, (st, en) in enumerate(ranges):
                    chunk_path = os.path.join(tmp, f"chunk-{n:03d}.pdf")
                    _write_page_range(path, st, en, chunk_path)
                    b, i, o = self._parse_one(chunk_path,
                                              extraction_prompt(st + 1, en))
                    _globalize_block_pages(b, st + 1, en)
                    blocks.extend(b)
                    in_tok += i
                    out_tok += o
        pages = len({p for p in (block_page_number(b) for b in blocks) if p is not None})
        rate_in = float(os.environ.get("TINKER_OPENAI_RATE_IN",
                                       self.RATES.get(self._model, (2.50, 10.00))[0]))
        rate_out = float(os.environ.get("TINKER_OPENAI_RATE_OUT",
                                        self.RATES.get(self._model, (2.50, 10.00))[1]))
        return ProviderResult(
            provider=self.name, model=self._model, blocks=blocks, pages=pages,
            cost_usd=_usd(in_tok, out_tok, rate_in, rate_out),
            cost_note="approximate (tokens x list rate)",
            input_tokens=in_tok, output_tokens=out_tok,
            raw_summary={"model": self._model, "chunks": len(ranges)})

    def _parse_one(self, chunk_path: str,
                   prompt: str) -> tuple[list[dict], int, int]:
        """One chat-completions extraction call.

        Returns (blocks, input_tokens, output_tokens).
        """
        with open(chunk_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        body = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "file", "file": {
                        "filename": os.path.basename(chunk_path),
                        "file_data": f"data:application/pdf;base64,{b64}"}},
                ],
            }],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.max_output_tokens(),
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
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(truncation_message(self._model, self.max_output_tokens()))
        try:
            content = choice["message"]["content"]
            payload = json.loads(_strip_fences(content))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI returned unparseable output: {exc}") from exc
        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)
        return normalize_llm_blocks(payload), in_tok, out_tok


class GeminiProvider(ParseProvider):
    """Multimodal LLM extraction via Gemini generateContent."""

    name = "gemini"
    # Approximate list rates ($ per 1M tokens); override via env.
    # gemini-2.0-flash was retired by Google; 3.8-flash is the current default.
    RATES = {"gemini-3.8-flash": (0.75, 3.75), "gemini-2.5-flash": (0.30, 2.50),
             "gemini-2.5-pro": (1.25, 10.00)}

    # Maximum output tokens per model; requests above the limit fail.
    MAX_OUTPUT = {"gemini-3.8-flash": 65_536, "gemini-2.5-flash": 65_536,
                  "gemini-2.5-pro": 65_536}
    DEFAULT_MAX_OUTPUT = 8_192

    # Input context is 1M tokens for current models; chunking is driven by
    # the output cap instead (MAX_PAGES_PER_CHUNK keeps each response well
    # under it), with the byte budget as a backstop for image-heavy pages.
    INPUT_LIMIT = 1_000_000
    CHUNK_FRACTION = 0.70
    TOKENS_PER_BYTE = 0.4
    MAX_PAGES_PER_CHUNK = 10

    default_model = "gemini-3.8-flash"
    model_env = "TINKER_GEMINI_PARSE_MODEL"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model)
        self.api_key = (os.environ.get("TINKER_GEMINI_API_KEY", "")
                        or os.environ.get("GOOGLE_API_KEY", ""))

    def max_output_tokens(self) -> int:
        return self.MAX_OUTPUT.get(self._model, self.DEFAULT_MAX_OUTPUT)

    def _chunk_byte_budget(self) -> int:
        return int(self.INPUT_LIMIT * self.CHUNK_FRACTION / self.TOKENS_PER_BYTE)

    @classmethod
    def configured(cls) -> tuple[bool, str]:
        key = os.environ.get("TINKER_GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        return (True, "") if key else (False, "Set TINKER_GEMINI_API_KEY to enable")

    @classmethod
    def available_models(cls) -> list[str]:
        return list(cls.RATES)

    def parse(self, path: str) -> ProviderResult:
        if not self.api_key:
            raise RuntimeError("TINKER_GEMINI_API_KEY not set")
        ranges = _chunk_ranges(path, self._chunk_byte_budget(),
                               self.MAX_PAGES_PER_CHUNK)
        blocks: list[dict] = []
        in_tok = out_tok = 0
        if len(ranges) == 1:
            b, i, o = self._parse_one(path, extraction_prompt())
            blocks, in_tok, out_tok = b, i, o
        else:
            import tempfile
            with tempfile.TemporaryDirectory(prefix="tinker-playground-") as tmp:
                for n, (st, en) in enumerate(ranges):
                    chunk_path = os.path.join(tmp, f"chunk-{n:03d}.pdf")
                    _write_page_range(path, st, en, chunk_path)
                    b, i, o = self._parse_one(chunk_path,
                                              extraction_prompt(st + 1, en))
                    _globalize_block_pages(b, st + 1, en)
                    blocks.extend(b)
                    in_tok += i
                    out_tok += o
        pages = len({p for p in (block_page_number(b) for b in blocks) if p is not None})
        rate_in = float(os.environ.get("TINKER_GEMINI_RATE_IN",
                                       self.RATES.get(self._model, (0.75, 3.75))[0]))
        rate_out = float(os.environ.get("TINKER_GEMINI_RATE_OUT",
                                        self.RATES.get(self._model, (0.75, 3.75))[1]))
        return ProviderResult(
            provider=self.name, model=self._model, blocks=blocks, pages=pages,
            cost_usd=_usd(in_tok, out_tok, rate_in, rate_out),
            cost_note="approximate (tokens x list rate)",
            input_tokens=in_tok, output_tokens=out_tok,
            raw_summary={"model": self._model, "chunks": len(ranges)})

    def _parse_one(self, chunk_path: str,
                   prompt: str) -> tuple[list[dict], int, int]:
        """One generateContent extraction call.

        Returns (blocks, input_tokens, output_tokens).
        """
        with open(chunk_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        body = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "application/pdf", "data": b64}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "maxOutputTokens": self.max_output_tokens()},
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
        cand = (data.get("candidates") or [{}])[0]
        if cand.get("finishReason") == "MAX_TOKENS":
            raise RuntimeError(truncation_message(self._model, self.max_output_tokens()))
        try:
            text = cand["content"]["parts"][0]["text"]
            payload = json.loads(_strip_fences(text))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            reason = cand.get("finishReason") or "unknown"
            raise RuntimeError(
                f"Gemini returned no parsable text (finishReason={reason})") from exc
        usage = data.get("usageMetadata") or {}
        in_tok = int(usage.get("promptTokenCount") or 0)
        out_tok = int(usage.get("candidatesTokenCount") or 0)
        return normalize_llm_blocks(payload), in_tok, out_tok


_PROVIDERS: dict[str, type[ParseProvider]] = {
    "extend": ExtendProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def provider_status() -> list[dict]:
    """For the Data tab: which providers can run right now, and their models."""
    out = []
    for name in PROVIDERS:
        cls = _PROVIDERS[name]
        ok, reason = cls.configured()
        try:
            models = cls.available_models()
            default = cls().model
        except Exception:
            models, default = [], ""
        out.append({"name": name, "configured": ok, "reason": reason,
                    "model": default, "models": models,
                    "default_model": default})
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
    # table_shape_rate is 1.0 when a provider found no tables at all, so only
    # providers that actually extracted tables can win "best tables".
    with_tables = [n for n in ok if (ok[n]["fidelity"].get("table_blocks") or 0) > 0]
    return {
        "coverage": max(ok, key=lambda n: cov(n) if cov(n) is not None else -1),
        "table_shape": (max(with_tables, key=lambda n: ok[n]["fidelity"].get("table_shape_rate") or 0)
                        if with_tables else None),
        "cheapest": min(ok, key=lambda n: ok[n]["cost_usd"]),
        "fastest": min(ok, key=lambda n: ok[n]["duration_s"]),
        # ok is None for checks that don't apply (no tables, not a scan);
        # only ok is False is a flag.
        "fewest_flags": min(ok, key=lambda n: sum(1 for c in ok[n]["checks"] if c.get("ok") is False)),
    }


def compare_providers(path: str,
                      selections: list[str] | dict[str, str] | None = None) -> dict:
    """Parse `path` with each selected provider and score them all.

    `selections` is None (every provider, default models), a list of
    provider names, or {provider_name: model} for per-provider models.
    """
    import datetime

    if selections is None:
        sel: dict[str, str | None] = {n: None for n in PROVIDERS}
    elif isinstance(selections, dict):
        sel = {n: m for n, m in selections.items() if n in _PROVIDERS}
    else:
        sel = {n: None for n in selections if n in _PROVIDERS}
    if not sel:
        sel = {n: None for n in PROVIDERS}
    scored: dict[str, dict] = {}
    for name, model in sel.items():
        provider = _PROVIDERS[name](model=model)
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
