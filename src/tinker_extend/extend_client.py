"""Thin wrapper over the Extend REST API (stdlib only, no SDK dependency).

Covers the calls the pipeline needs: file upload, synchronous parse, and
asynchronous parse runs with polling (the production path — the sync endpoint
has a 5-minute timeout).

Docs: https://docs.extend.ai/api-quickstart (API version 2026-02-09).
"""
from __future__ import annotations

import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid

from . import __version__


class ExtendAPIError(RuntimeError):
    """An HTTP-level API error."""

    def __init__(self, http_status: int, code: str, message: str, retryable: bool):
        super().__init__(f"Extend API HTTP {http_status} [{code}]: {message}")
        self.http_status = http_status
        self.code = code
        self.retryable = retryable


class ExtendRunFailed(RuntimeError):
    """A parse run finished with status FAILED."""

    def __init__(self, failure_reason: str | None, failure_message: str | None):
        super().__init__(f"Extend parse run failed [{failure_reason}]: {failure_message}")
        self.failure_reason = failure_reason
        self.failure_message = failure_message


def encode_multipart(field_name: str, filename: str, data: bytes) -> tuple[bytes, str]:
    """Encode a single-file multipart/form-data body. Returns (body, content_type)."""
    boundary = f"----tinkerextend{uuid.uuid4().hex}"
    buf = io.BytesIO()
    buf.write(f"--{boundary}\r\n".encode())
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    buf.write(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
    )
    buf.write(f"Content-Type: {mime}\r\n\r\n".encode())
    buf.write(data)
    buf.write(f"\r\n--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


class ExtendClient:
    def __init__(
        self,
        api_key: str,
        api_version: str = "2026-02-09",
        base_url: str = "https://api.extend.ai",
        workspace_id: str | None = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.api_version = api_version
        self.base_url = base_url.rstrip("/")
        self.workspace_id = workspace_id

    # -- low-level ---------------------------------------------------------
    def _headers(self, content_type: str | None = "application/json") -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-extend-api-version": self.api_version,
            "User-Agent": f"tinker-extend/{__version__}",
        }
        if content_type:
            headers["Content-Type"] = content_type
        if self.workspace_id:
            headers["x-extend-workspace-id"] = self.workspace_id
        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        raw_body: bytes | None = None,
        content_type: str | None = "application/json",
    ) -> dict:
        url = f"{self.base_url}{path}"
        body = raw_body if raw_body is not None else (
            json.dumps(payload).encode() if payload is not None else None
        )
        req = urllib.request.Request(url, data=body, method=method, headers=self._headers(content_type))
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
                return json.loads(raw.decode()) if raw else {}
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read().decode())
            except Exception:
                err = {}
            raise ExtendAPIError(
                exc.code,
                err.get("code", "http_error"),
                err.get("message", f"HTTP {exc.code}"),
                bool(err.get("retryable", exc.code in (429,) or 500 <= exc.code < 600)),
            ) from exc

    # -- files -------------------------------------------------------------
    def upload_file(self, path: str) -> str:
        """Upload a local file; returns the Extend file id."""
        with open(path, "rb") as f:
            data = f.read()
        body, content_type = encode_multipart("file", os.path.basename(path), data)
        resp = self._request("POST", "/files/upload", raw_body=body, content_type=content_type)
        file_id = resp.get("id")
        if not file_id:
            raise ExtendAPIError(200, "missing_file_id", f"Upload response had no id: {resp}", False)
        return file_id

    # -- parse -------------------------------------------------------------
    @staticmethod
    def _check_run(run: dict) -> dict:
        status = run.get("status")
        if status == "FAILED":
            raise ExtendRunFailed(run.get("failureReason"), run.get("failureMessage"))
        return run

    def parse_sync(self, file: dict) -> dict:
        """POST /parse — synchronous; 5-minute server timeout. For small docs/tests."""
        run = self._request("POST", "/parse", {"file": file})
        return self._check_run(run)

    def parse_async(
        self,
        file: dict,
        poll_interval_s: float = 5.0,
        timeout_s: float = 1800.0,
        metadata: dict | None = None,
        engine: str | None = None,
    ) -> dict:
        """POST /parse_runs then poll until PROCESSED/FAILED. The production path.

        engine: "parse_light" (cheap, text-heavy docs) or "parse_performance"
        (tables, scans, handwriting). None = provider default.
        """
        body: dict = {"file": file}
        if metadata:
            body["metadata"] = metadata
        if engine:
            body["config"] = {"engine": engine}
        run = self._request("POST", "/parse_runs", body)
        run_id = run.get("id")
        if not run_id:
            raise ExtendAPIError(200, "missing_run_id", f"parse_runs response had no id: {run}", False)

        deadline = time.monotonic() + timeout_s
        while True:
            run = self._request("GET", f"/parse_runs/{run_id}")
            status = run.get("status")
            if status == "PROCESSED":
                return run
            if status == "FAILED":
                raise ExtendRunFailed(run.get("failureReason"), run.get("failureMessage"))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Parse run {run_id} did not finish within {timeout_s}s")
            time.sleep(poll_interval_s)
