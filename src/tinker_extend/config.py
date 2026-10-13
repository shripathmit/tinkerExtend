"""Configuration from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    api_key: str
    api_version: str = "2026-02-09"
    base_url: str = "https://api.extend.ai"
    workspace_id: str | None = None
    kb_path: str = "./kb.sqlite3"
    # Async parse runs: how long to wait between polls, and the overall cap.
    poll_interval_s: float = 5.0
    poll_timeout_s: float = 1800.0


def load_config(kb_path: str | None = None) -> Config:
    api_key = os.environ.get("EXTEND_API_KEY", "").strip()
    if not api_key or api_key == "REPLACE_ME":
        raise RuntimeError(
            "EXTEND_API_KEY is not set. Create a key at "
            "dashboard.extend.ai (Developers page) and export EXTEND_API_KEY."
        )
    return Config(
        api_key=api_key,
        api_version=os.environ.get("EXTEND_API_VERSION", "2026-02-09"),
        base_url=os.environ.get("EXTEND_BASE_URL", "https://api.extend.ai").rstrip("/"),
        workspace_id=os.environ.get("EXTEND_WORKSPACE_ID") or None,
        kb_path=kb_path or os.environ.get("TINKER_KB_PATH", "./kb.sqlite3"),
    )
