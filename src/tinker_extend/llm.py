"""Optional generative layer: OpenAI-compatible chat completions (stdlib only).

Configure via environment; when absent, the app serves extractive answers.
  TINKER_LLM_API_KEY   - required to enable generation
  TINKER_LLM_BASE_URL  - default https://api.openai.com/v1 (any compatible endpoint works)
  TINKER_LLM_MODEL     - default gpt-4o-mini
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_s: float = 90.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {str(data)[:300]}") from exc


def llm_from_env() -> LLMClient | None:
    api_key = os.environ.get("TINKER_LLM_API_KEY", "").strip()
    if not api_key:
        return None
    return LLMClient(
        api_key=api_key,
        base_url=os.environ.get("TINKER_LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("TINKER_LLM_MODEL", "gpt-4o-mini"),
    )
