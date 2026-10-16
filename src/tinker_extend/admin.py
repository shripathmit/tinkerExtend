"""Admin API behind the access code: ingestion waves, parse fidelity,
re-parse, golden sets and dead letters. The UI lives in static/ and calls
these endpoints; login/session endpoints here are public.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

from .config import load_config
from .embeddings import embedder_from_env
from .extend_client import ExtendClient
from .chunking import TAG_STRIP_RE
from .fidelity import (
    _is_garbled, _table_shape_ok, check_golden_doc, fidelity_checks,
    fidelity_metrics, fidelity_verdict, load_golden_docs, red_flags, table_grid,
)
from .ingest import SUPPORTED_EXTENSIONS, iter_candidate_files, sha256_file
from .pages import resolve_pdf
from .pipeline import ENGINE_RATES, WaveIngester
from .store import KnowledgeBase

# Login/session routes are public; every /api/admin route is
# registered on _api, which requires the admin code. _api is mounted onto
# router at the bottom of this module, after all its routes exist.
router = APIRouter()

KB_PATH = os.environ.get("TINKER_KB_PATH", "./kb.sqlite3")
SEED_DIR = os.environ.get("TINKER_SEED_DIR", "./seed_docs")
UPLOAD_DIR = os.environ.get("TINKER_UPLOAD_DIR", "./uploads")
PAGE_IMG_DIR = os.environ.get("TINKER_PAGE_IMG_DIR", "./page_images")
MAX_UPLOAD_BYTES = 150 * 1024 * 1024  # per file
PLAYGROUND_DIR = os.environ.get("TINKER_PLAYGROUND_DIR", "/app/data/playground")

# Shared access code for the admin console. Override in production.
ADMIN_CODE = os.environ.get("TINKER_ADMIN_CODE", "extend")
ADMIN_COOKIE = "tinker_admin"
ADMIN_COOKIE_MAX_AGE = 7 * 24 * 3600


def _session_token() -> str:
    """Cookie value: an HMAC of the code, so the code itself never sits in the cookie."""
    return hmac.new(ADMIN_CODE.encode(), b"tinker-admin-session",
                    hashlib.sha256).hexdigest()


def is_admin(request: Request) -> bool:
    """Signed-in via the console cookie, or an X-Admin-Code header (scripts)."""
    cookie = request.cookies.get(ADMIN_COOKIE, "")
    header = request.headers.get("x-admin-code", "")
    return (hmac.compare_digest(cookie, _session_token())
            or hmac.compare_digest(header, ADMIN_CODE))


def require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(status_code=401, detail="admin code required")


_api = APIRouter(dependencies=[Depends(require_admin)])


def _resolve_eval_file(env_key: str, default: str) -> str:
    """Golden-set paths: explicit env, else ./eval, else beside the package."""
    p = os.environ.get(env_key)
    if p and os.path.exists(p):
        return p
    if os.path.exists(default):
        return default
    # Fall back to the repo checkout that contains this package
    # (src/tinker_extend -> repo root).
    alt = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", default.lstrip("./")))
    return alt if os.path.exists(alt) else default


GOLDEN_DOCS = _resolve_eval_file("TINKER_GOLDEN_DOCS", "./eval/golden_docs.json")
GOLDEN_QA = _resolve_eval_file("TINKER_GOLDEN_QA", "./eval/golden_10q.jsonl")

# Live background wave jobs: wave name -> {status, progress, summary}.
JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _kb() -> KnowledgeBase:
    return KnowledgeBase(KB_PATH)


def _metrics_from_row(r: dict) -> dict:
    return {
        "num_pages": r["num_pages"], "num_chunks": r["num_chunks"],
        "empty_chunks": r["empty_chunks"],
        "empty_rate": (r["empty_chunks"] / r["num_chunks"]) if r["num_chunks"] else 0,
        "garbled_chunks": r["garbled_chunks"],
        "garbled_rate": (r["garbled_chunks"] / r["num_chunks"]) if r["num_chunks"] else 0,
        "table_blocks": r["num_tables"],
        "table_shape_ok": r["table_shape_ok"],
        "table_shape_rate": (r["table_shape_ok"] / r["num_tables"]) if r["num_tables"] else 1.0,
        "ocr_confidence": r["ocr_confidence"],
        "repeated_ngrams": r["repeated_ngrams"],
        "text_coverage": r["text_coverage"],
    }


def _client_factory() -> ExtendClient:
    cfg = load_config()
    return ExtendClient(api_key=cfg.api_key, api_version=cfg.api_version,
                        base_url=cfg.base_url, workspace_id=cfg.workspace_id)


def _run_wave_job(wave: str, paths: list[str], workers: int,
                  force: bool = False, engine: str | None = None,
                  doc_ids: list[str] | None = None) -> None:
    with _JOBS_LOCK:
        JOBS[wave] = {"status": "running", "progress": {"total": len(paths)},
                      "started_at": time.time(), "kind": "reparse" if force else "wave",
                      "doc_ids": doc_ids or []}
    try:
        ingester = WaveIngester(_client_factory, KB_PATH, workers=workers,
                                embedder=embedder_from_env(), force=force,
                                engine_override=engine)

        def on_progress(progress: dict, res: dict) -> None:
            with _JOBS_LOCK:
                JOBS[wave]["progress"] = progress
                JOBS[wave]["last"] = {"source": os.path.basename(res["source"]),
                                     "status": res["status"]}

        summary = ingester.run(paths, wave, progress_cb=on_progress)
        summary.pop("results", None)
        with _JOBS_LOCK:
            JOBS[wave] = {**JOBS[wave], "status": "done", "progress": summary}
    except Exception as exc:  # noqa: BLE001
        with _JOBS_LOCK:
            JOBS[wave]["status"] = f"failed: {exc}"


@_api.get("/api/admin/overview")
def api_overview():
    kb = _kb()
    try:
        waves = kb.waves()
        recent = kb.telemetry(8)
        flags = []
        for r in kb.telemetry(200):
            fl = red_flags(_metrics_from_row(r))
            if fl:
                flags.append({"doc": os.path.basename(r["source"]),
                             "doc_id": r["doc_id"], "engine": r["engine"],
                             "doc_type": r["doc_type"], "flags": fl})
        with _JOBS_LOCK:
            jobs = dict(JOBS)
        vectors = kb.embedding_stats(embedder_from_env().model)
        return {"overview": kb.overview(), "vectors": vectors, "waves": waves,
                "recent": recent, "red_flags": flags[:20], "jobs": jobs}
    finally:
        kb.close()


@_api.get("/api/admin/telemetry")
def api_telemetry(limit: int = 100):
    kb = _kb()
    try:
        rows = kb.telemetry(limit)
        for r in rows:
            r["name"] = os.path.basename(r["source"])
            r["flags"] = red_flags(_metrics_from_row(r))
        return {"rows": rows}
    finally:
        kb.close()


@_api.get("/api/admin/fidelity")
def api_fidelity():
    kb = _kb()
    try:
        by_type = kb.fidelity_by_type()
        worst = []
        for r in kb.telemetry(200):
            fl = red_flags(_metrics_from_row(r))
            if fl:
                worst.append({"doc": os.path.basename(r["source"]),
                             "doc_id": r["doc_id"],
                             "doc_type": r["doc_type"], "flags": fl})
        return {"by_type": by_type, "worst": worst[:25]}
    finally:
        kb.close()


def golden_doc_checks(kb: KnowledgeBase) -> list[dict]:
    """check_golden_doc for every spec in GOLDEN_DOCS (status not_ingested
    when the document isn't in the KB)."""
    if not os.path.exists(GOLDEN_DOCS):
        return []
    docs = {os.path.basename(d["source"]): d["doc_id"]
            for d in (dict(r) for r in
                      kb.conn.execute("SELECT doc_id, source FROM documents"))}
    hashes = {d["file_hash"]: d["doc_id"]
              for d in (dict(r) for r in
                        kb.conn.execute("SELECT doc_id, file_hash FROM documents"))}
    results = []
    for spec in load_golden_docs(GOLDEN_DOCS):
        doc_id = docs.get(spec.get("doc", ""))
        if not doc_id:
            # The spec may name the seed copy; match by content hash.
            for cand in (spec.get("doc", ""),
                         os.path.join(SEED_DIR, spec.get("doc", ""))):
                if cand and os.path.exists(cand):
                    try:
                        doc_id = hashes.get(sha256_file(cand))
                    except OSError:
                        doc_id = None
                    if doc_id:
                        break
        if not doc_id:
            results.append({**spec, "status": "not_ingested", "ok": False,
                            "probes_passed": 0, "probes_total": 0,
                            "probes": [], "tables": spec.get("tables", [])})
            continue
        text = "\n".join(r["content"] for r in kb.conn.execute(
            "SELECT content FROM chunks WHERE doc_id=?", (doc_id,)))
        results.append({**check_golden_doc(spec, text), "status": "checked"})
    return results


@_api.get("/api/admin/golden_docs")
def api_golden_docs():
    if not os.path.exists(GOLDEN_DOCS):
        return {"specs": [], "note": "no golden docs defined yet"}
    kb = _kb()
    try:
        return {"specs": golden_doc_checks(kb)}
    finally:
        kb.close()


def run_golden_suite(kb: KnowledgeBase) -> dict:
    """Run every golden assertion plus the keyword-vs-hybrid comparison."""
    from .eval import compare_modes, golden_assertions, load_eval_set
    cases = load_eval_set(GOLDEN_QA) if os.path.exists(GOLDEN_QA) else []
    embedder = embedder_from_env()
    result = golden_assertions(kb, cases, golden_doc_checks(kb), kb.list_feedback(500),
                               embedder=embedder, k=5)
    result["compare"] = compare_modes(kb, cases, embedder, k=5) if cases else None
    result["ran_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["chunks"] = kb.stats()["chunks"]
    result["vectors_rev"] = kb.vectors_rev()
    kb.set_meta("golden_last_run", json.dumps(result))
    return result


@_api.get("/api/admin/golden")
def api_golden():
    """Last stored golden-suite run (runs it once if none stored yet)."""
    kb = _kb()
    try:
        raw = kb.get_meta("golden_last_run")
        last = json.loads(raw) if raw else None
        # Stale once chunks/vectors changed (ingest, re-parse) since the run.
        if not last or last.get("vectors_rev") != kb.vectors_rev():
            last = run_golden_suite(kb)
        return last
    finally:
        kb.close()


@_api.post("/api/admin/golden/run")
def api_golden_run():
    kb = _kb()
    try:
        return run_golden_suite(kb)
    finally:
        kb.close()


@_api.post("/api/admin/feedback/{fid}/resolve")
def api_feedback_resolve(fid: int):
    kb = _kb()
    try:
        if not kb.resolve_feedback(fid):
            return JSONResponse({"error": "not found"}, status_code=404)
        kb.set_meta("golden_last_run", "")
        return {"ok": True}
    finally:
        kb.close()


@_api.get("/api/admin/golden_qa")
def api_golden_qa():
    from .eval import compare_modes, load_eval_set
    if not os.path.exists(GOLDEN_QA):
        return {"cases": 0, "note": "no golden Q&A yet"}
    kb = _kb()
    try:
        cases = load_eval_set(GOLDEN_QA)
        return compare_modes(kb, cases, embedder_from_env(), k=5)
    finally:
        kb.close()


@_api.get("/api/admin/dead_letters")
def api_dead_letters():
    kb = _kb()
    try:
        return {"rows": kb.list_dead_letters()}
    finally:
        kb.close()


@_api.post("/api/admin/dead_letters/{dl_id}/resolve")
def api_dl_resolve(dl_id: int):
    kb = _kb()
    try:
        kb.resolve_dead_letter(dl_id)
        return {"ok": True}
    finally:
        kb.close()


@_api.post("/api/admin/dead_letters/{dl_id}/retry")
def api_dl_retry(dl_id: int):
    kb = _kb()
    try:
        row = kb.requeue_dead_letter(dl_id)
    finally:
        kb.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    if not os.path.exists(row["source"]):
        return JSONResponse({"error": f"source file missing: {row['source']}"},
                            status_code=400)
    wave = f"retry-{int(time.time())}"
    t = threading.Thread(target=_run_wave_job, args=(wave, [row["source"]], 1),
                         daemon=True)
    t.start()
    return {"ok": True, "wave": wave}


@_api.post("/api/admin/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    """Upload documents from the operator's machine into the upload dir.

    Files land under TINKER_UPLOAD_DIR (put it on the persistent volume
    in production); a wave can then ingest them. Filenames are sanitized,
    extensions restricted to the ingest-supported set.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved, skipped = [], []
    for f in files:
        name = os.path.basename(f.filename or "").strip()
        ext = os.path.splitext(name)[1].lower()
        if not name or ext not in SUPPORTED_EXTENSIONS:
            skipped.append({"file": f.filename, "reason": "unsupported type"})
            continue
        dest = os.path.join(UPLOAD_DIR, name)
        stem, suffix = os.path.splitext(name)
        i = 1
        while os.path.exists(dest):
            dest = os.path.join(UPLOAD_DIR, f"{stem}-{i}{suffix}")
            i += 1
        size = 0
        try:
            with open(dest, "wb") as out:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise ValueError("file too large")
                    out.write(chunk)
        except ValueError:
            if os.path.exists(dest):
                os.remove(dest)
            skipped.append({"file": name, "reason": "over 150MB"})
            continue
        finally:
            await f.close()
        saved.append(os.path.basename(dest))
    return {"ok": True, "dir": os.path.abspath(UPLOAD_DIR),
            "saved": saved, "skipped": skipped}


@_api.get("/api/admin/uploads")
def api_uploads():
    if not os.path.isdir(UPLOAD_DIR):
        return {"dir": os.path.abspath(UPLOAD_DIR), "files": []}
    return {"dir": os.path.abspath(UPLOAD_DIR),
            "files": sorted(os.listdir(UPLOAD_DIR))}


def _run_provider_comparison(run_id: int, path: str,
                             selections: dict[str, str]) -> None:
    """Background thread: run the provider bake-off, store the result."""
    kb = _kb()
    try:
        from .providers import compare_providers
        result = compare_providers(path, selections)
        kb.finish_provider_run(run_id, result, status="done")
    except Exception as exc:  # noqa: BLE001 - the row must record the failure
        kb.finish_provider_run(
            run_id, {"error": f"{type(exc).__name__}: {exc}"}, status="error")
    finally:
        kb.close()


@_api.get("/api/admin/providers")
def api_providers():
    """Beta playground: which parse providers are configured right now."""
    from .providers import provider_status
    return {"providers": provider_status(),
            "beta": True,
            "note": "Experimental playground. Production ingestion stays on Extend."}


@_api.post("/api/admin/provider_runs")
async def api_provider_run_start(file: UploadFile = File(...),
                                 providers: str = Form("[]"),
                                 models: str = Form("{}")):
    """Upload a PDF and start a parse-provider bake-off in the background.

    `providers` is a JSON list of provider names; `models` an optional
    JSON {provider: model} for per-provider model selection.
    Returns immediately; poll GET /api/admin/provider_runs/{id} for the
    result. The playground never touches the production knowledge base.
    """
    from .providers import PROVIDERS
    try:
        names = [n for n in json.loads(providers or "[]") if n in PROVIDERS]
    except (json.JSONDecodeError, TypeError):
        names = []
    if not names:
        names = list(PROVIDERS)
    try:
        model_sel = json.loads(models or "{}")
    except (json.JSONDecodeError, TypeError):
        model_sel = {}
    selections = {n: model_sel.get(n) for n in names}
    name = os.path.basename(file.filename or "").strip()
    if not name.lower().endswith(".pdf"):
        return JSONResponse({"error": "playground accepts PDF files only"},
                            status_code=400)
    os.makedirs(PLAYGROUND_DIR, exist_ok=True)
    dest = os.path.join(PLAYGROUND_DIR, name)
    stem, suffix = os.path.splitext(name)
    i = 1
    while os.path.exists(dest):
        dest = os.path.join(PLAYGROUND_DIR, f"{stem}-{i}{suffix}")
        i += 1
    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError("file too large")
                out.write(chunk)
    except ValueError:
        if os.path.exists(dest):
            os.remove(dest)
        return JSONResponse({"error": "file too large"}, status_code=400)
    # Defensive: never hand an empty or non-PDF file to the providers.
    # (A missing write once shipped empty files here; fail loudly instead.)
    try:
        with open(dest, "rb") as f:
            head = f.read(1024)
    except OSError:
        head = b""
    if not head or b"%PDF-" not in head:
        os.remove(dest)
        return JSONResponse({"error": "upload failed: file is empty or not a PDF"},
                            status_code=400)
    kb = _kb()
    try:
        run_id = kb.create_provider_run(os.path.basename(dest), dest, names)
    finally:
        kb.close()
    t = threading.Thread(target=_run_provider_comparison,
                         args=(run_id, dest, selections), daemon=True)
    t.start()
    return {"id": run_id, "status": "running", "providers": names,
            "models": selections}


@_api.get("/api/admin/provider_runs")
def api_provider_runs():
    kb = _kb()
    try:
        return {"runs": kb.list_provider_runs()}
    finally:
        kb.close()


@_api.get("/api/admin/provider_runs/{run_id}")
def api_provider_run(run_id: int):
    kb = _kb()
    try:
        row = kb.get_provider_run(run_id)
    finally:
        kb.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    return row


def recompute_doc_telemetry(kb: KnowledgeBase, doc_id: str) -> dict | None:
    """Recompute fidelity metrics from stored blocks (no API calls).

    Used when the metric code changes or a row was recorded with a bug.
    Wave/engine/cost/duration are preserved.
    """
    doc = kb.conn.execute(
        "SELECT doc_id, source, file_hash FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    if not doc:
        return None
    blocks: list = []
    chunks: list = []
    for r in kb.conn.execute(
            "SELECT content, blocks_json FROM chunks WHERE doc_id=?", (doc_id,)):
        chunks.append(r["content"])
        try:
            blocks.extend(json.loads(r["blocks_json"] or "[]"))
        except (json.JSONDecodeError, TypeError):
            pass
    metrics = fidelity_metrics(blocks, chunks)
    # Coverage when the source file is still around (uploads dir / seed dir).
    src = _source_path(dict(doc))
    if src:
        try:
            from pypdf import PdfReader
            text = "\n".join((p.extract_text() or "") for p in PdfReader(src).pages)
            if text.strip():
                from .fidelity import text_coverage as _tc
                metrics["text_coverage"] = _tc(text, "\n".join(chunks))
        except Exception:
            pass
    if "text_coverage" not in metrics:
        # Source PDF not on this server: keep the coverage measured at ingest
        # rather than erasing it.
        prev = kb.telemetry_row(doc_id)
        if prev and prev.get("text_coverage") is not None:
            metrics["text_coverage"] = prev["text_coverage"]
    kb.update_telemetry_metrics(doc_id, metrics)
    return {"doc_id": doc_id, "metrics": metrics,
            "flags": red_flags(metrics)}


@_api.post("/api/admin/telemetry/recompute")
def api_telemetry_recompute(payload: dict):
    doc_id = (payload or {}).get("doc_id")
    kb = _kb()
    try:
        if doc_id:
            res = recompute_doc_telemetry(kb, doc_id)
            if not res:
                return JSONResponse({"error": "doc not found"}, status_code=404)
            return {"ok": True, "docs": [res]}
        out = []
        for r in kb.conn.execute("SELECT doc_id FROM parse_telemetry"):
            res = recompute_doc_telemetry(kb, r["doc_id"])
            if res:
                out.append(res)
        return {"ok": True, "docs": out}
    finally:
        kb.close()


@_api.post("/api/admin/waves")
def api_start_wave(payload: dict):
    try:
        _client_factory()
    except RuntimeError as exc:
        return JSONResponse({"error": f"cannot start wave: {exc}"}, status_code=400)
    source_dir = payload.get("dir") or SEED_DIR
    wave = (payload.get("wave") or "").strip()
    limit = int(payload.get("limit") or 0)
    workers = int(payload.get("workers") or 4)
    if not wave:
        return JSONResponse({"error": "wave name is required"}, status_code=400)
    if not os.path.isdir(source_dir):
        return JSONResponse({"error": f"not a directory: {source_dir}"}, status_code=400)
    paths = iter_candidate_files([source_dir])
    if limit:
        paths = paths[:limit]
    if not paths:
        return JSONResponse({"error": "no documents found"}, status_code=400)
    with _JOBS_LOCK:
        if JOBS.get(wave, {}).get("status") == "running":
            return JSONResponse({"error": f"wave {wave} already running"},
                                status_code=409)
    t = threading.Thread(target=_run_wave_job, args=(wave, paths, workers),
                         daemon=True)
    t.start()
    return {"ok": True, "wave": wave, "files": len(paths)}


@_api.get("/api/admin/jobs")
def api_jobs():
    with _JOBS_LOCK:
        return {"jobs": dict(JOBS)}


_EDGE_NUM_RE = re.compile(r"^\d{1,4}$")


# Issue kinds in display order, with the key the UI uses for icons/help.
ISSUE_KINDS = [
    ("table_shape", "Table shape inconsistent"),
    ("page_leak", "Page number leaked into body"),
    ("garbled", "Garbled text"),
    ("empty", "Empty passage"),
]


def _plain_lines(text: str) -> list[str]:
    """Non-empty lines of a chunk with markup stripped, whitespace collapsed."""
    out = []
    for line in (text or "").splitlines():
        t = " ".join(TAG_STRIP_RE.sub(" ", line).split())
        if t:
            out.append(t)
    return out


def _context_lines(lines: list[str], mark: int, keep: int = 3) -> list[dict]:
    """The marked line plus `keep` lines of context, with gap markers."""
    lo, hi = max(0, mark - keep), min(len(lines), mark + keep + 1)
    out: list[dict] = []
    if lo > 0:
        out.append({"gap": lo})
    out += [{"n": i + 1, "text": lines[i][:240], "mark": i == mark} for i in range(lo, hi)]
    if hi < len(lines):
        out.append({"gap": len(lines) - hi})
    return out


def _problem_groups(kb: KnowledgeBase, doc_id: str, per_kind: int = 3) -> list[dict]:
    """Chunks that show why a document is flagged, grouped by issue.

    Each group: {key, kind, total, items[:per_kind]}. Items carry the chunk,
    its page, a short plain-text excerpt, and issue-specific detail: the
    parsed table grid with mismatched rows (table_shape), or the passage's
    lines with the leaked page number marked (page_leak).
    """
    groups = {key: {"key": key, "kind": label, "total": 0, "items": []}
              for key, label in ISSUE_KINDS}

    def add(key: str, row, extra: dict | None = None, text: str | None = None) -> None:
        g = groups[key]
        g["total"] += 1
        if len(g["items"]) >= per_kind:
            return
        try:
            pages = json.loads(row["pages_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            pages = []
        plain = " ".join(TAG_STRIP_RE.sub(" ", text if text is not None else row["content"]).split())
        g["items"].append({"kind": g["kind"], "chunk_id": row["chunk_id"],
                           "page": pages[0] if pages else None,
                           "text": plain[:320] + ("…" if len(plain) > 320 else ""),
                           **(extra or {})})

    rows = kb.conn.execute(
        "SELECT chunk_id, content, blocks_json, pages_json FROM chunks"
        " WHERE doc_id = ? ORDER BY ordinal", (doc_id,)).fetchall()
    parsed = [(r, json.loads(r["blocks_json"] or "[]")) for r in rows]
    page_nums: set[str] = set()
    for _, blocks in parsed:
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "page_number":
                t = TAG_STRIP_RE.sub(" ", b.get("content") or "").strip()
                if _EDGE_NUM_RE.match(t):
                    page_nums.add(t)

    for r, blocks in parsed:
        for b in blocks:
            if not (isinstance(b, dict) and b.get("type") == "table"):
                continue
            content = b.get("content") or ""
            if _table_shape_ok(content):
                continue
            grid = table_grid(content)
            # Rows around the first bad one, so the table itself can be
            # shown with the short/long rows marked.
            first_bad = grid["bad_rows"][0] if grid["bad_rows"] else 0
            lo = max(0, first_bad - 4)
            add("table_shape", r, {"table": {
                "rows": [{**row, "bad": (lo + i) in grid["bad_rows"], "index": lo + i + 1}
                         for i, row in enumerate(grid["rows"][lo:lo + 10])],
                "expected_width": grid["expected_width"],
                "bad_count": len(grid["bad_rows"]),
                "total_rows": len(grid["rows"]),
            }}, text=content)

    for r, _ in parsed:
        lines = _plain_lines(r["content"])
        if not lines:
            continue
        for idx, edge in ((0, "start"), (len(lines) - 1, "end")):
            if lines[idx] in page_nums:
                add("page_leak", r, {"leak": {
                    "value": lines[idx], "edge": edge,
                    "lines": _context_lines(lines, idx),
                }})
                break

    for r, _ in parsed:
        if _is_garbled(r["content"]):
            add("garbled", r)
        elif len(r["content"].strip()) < 20:
            add("empty", r)
    return [g for g in groups.values() if g["total"]]


def _source_path(doc: dict) -> str | None:
    """The original file for a document, verified by content hash.

    Checks the stored path, then the seed and upload dirs. A candidate is
    only accepted if its hash is the doc_id, so a re-parse can never pick
    up a different file.
    """
    candidates = [doc["source"]]
    for d in (SEED_DIR, UPLOAD_DIR):
        p = resolve_pdf(doc["source"], d)
        if p:
            candidates.append(p)
        cand = os.path.join(d, os.path.basename(doc["source"] or ""))
        candidates.append(cand)
    for c in candidates:
        if c and os.path.isfile(c):
            try:
                if sha256_file(c) == doc["file_hash"]:
                    return c
            except OSError:
                continue
    return None


@_api.get("/api/admin/doc/{doc_id}/fidelity")
def api_doc_fidelity(doc_id: str, per_kind: int = 3):
    kb = _kb()
    try:
        doc = kb.get_document(doc_id)
        if not doc:
            return JSONResponse({"error": "doc not found"}, status_code=404)
        row = kb.telemetry_row(doc_id)
        metrics = _metrics_from_row(row) if row else {}
        pages = (row or {}).get("num_pages") or doc.get("num_pages") or 0
        return {
            "doc_id": doc_id,
            "name": os.path.basename(doc["source"]),
            "checks": fidelity_checks(metrics) if row else [],
            "verdict": (fidelity_verdict(metrics, pages, ENGINE_RATES["parse_performance"])
                        if row else "No fidelity metrics recorded for this document yet."),
            "doc_type": (row or {}).get("doc_type", ""),
            "engine": (row or {}).get("engine", ""),
            "pages": pages,
            "metrics": metrics,
            "flags": red_flags(metrics) if row else [],
            "groups": (groups := _problem_groups(kb, doc_id, per_kind=max(1, min(per_kind, 25)))),
            "excerpts": [i for g in groups for i in g["items"]][:3],
            "num_chunks": (row or {}).get("num_chunks") or doc.get("num_chunks") or 0,
            "estimated_cost_usd": round(pages * ENGINE_RATES["parse_performance"], 4),
            "source_available": _source_path(doc) is not None,
            "history": kb.reparse_history(doc_id),
        }
    finally:
        kb.close()


@_api.post("/api/admin/reparse")
def api_reparse(payload: dict):
    """Re-parse documents with the Performance engine (spends Extend credits)."""
    try:
        _client_factory()
    except RuntimeError as exc:
        return JSONResponse({"error": f"cannot re-parse: {exc}"}, status_code=400)
    doc_ids = [d for d in (payload or {}).get("doc_ids") or [] if isinstance(d, str)]
    engine = (payload or {}).get("engine") or "parse_performance"
    if engine not in ENGINE_RATES:
        return JSONResponse({"error": f"unknown engine {engine}"}, status_code=400)
    if not doc_ids:
        return JSONResponse({"error": "doc_ids is required"}, status_code=400)
    kb = _kb()
    try:
        paths, missing = [], []
        for doc_id in doc_ids:
            doc = kb.get_document(doc_id)
            path = _source_path(doc) if doc else None
            if path:
                paths.append(path)
            else:
                missing.append(os.path.basename(doc["source"]) if doc else doc_id)
    finally:
        kb.close()
    if missing:
        return JSONResponse({"error": "source file not available for: " + ", ".join(missing)},
                            status_code=400)
    job = f"reparse-{int(time.time())}"
    threading.Thread(target=_run_wave_job, args=(job, paths, min(4, len(paths))),
                     kwargs={"force": True, "engine": engine, "doc_ids": doc_ids},
                     daemon=True).start()
    return {"ok": True, "job": job, "files": len(paths)}


# -- session (public) ------------------------------------------------------
def _with_session(resp, signed_in: bool):
    if signed_in:
        resp.set_cookie(ADMIN_COOKIE, _session_token(), max_age=ADMIN_COOKIE_MAX_AGE,
                        httponly=True, samesite="lax")
    else:
        resp.delete_cookie(ADMIN_COOKIE)
    return resp


@_api.delete("/api/admin/documents/{doc_id}")
def api_delete_document(doc_id: str):
    """Permanently delete a document and all its derived data.

    Removes the documents row (cascades to chunks and parse telemetry),
    FTS entries, reparse history, rendered page images, and the source
    upload when it lives inside the upload dir. Bumps the vectors revision
    so cached embeddings and golden runs invalidate. Cannot be undone.
    """
    kb = _kb()
    try:
        row = kb.conn.execute(
            "SELECT doc_id, source FROM documents WHERE doc_id = ?",
            (doc_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")
        source = row["source"] or ""
        name = os.path.basename(source)
        chunks = kb.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()[0]
        kb.delete_document(doc_id)
    finally:
        kb.close()
    removed = []
    img_dir = os.path.join(PAGE_IMG_DIR, doc_id)
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir, ignore_errors=True)
        removed.append("page images")
    try:
        up = os.path.realpath(UPLOAD_DIR)
        sp = os.path.realpath(source)
        if sp.startswith(up + os.sep) and os.path.isfile(sp):
            os.remove(sp)
            removed.append("source file")
    except OSError:
        pass
    return {"deleted": True, "doc_id": doc_id, "name": name,
            "chunks": chunks, "removed": removed}


@router.get("/api/session")
def api_session(request: Request):
    return {"admin": is_admin(request)}


@router.post("/api/admin/login")
def api_admin_login(payload: dict):
    code = str((payload or {}).get("code") or "").strip()
    if not hmac.compare_digest(code, ADMIN_CODE):
        return JSONResponse({"error": "Wrong code."}, status_code=401)
    return _with_session(JSONResponse({"ok": True}), True)


@router.post("/api/admin/logout")
def api_admin_logout():
    return _with_session(JSONResponse({"ok": True}), False)


@router.get("/admin")
def admin_index():
    """The console is now the unified app's locked tabs."""
    return RedirectResponse("/#overview", status_code=307)


@router.post("/admin/login")
def admin_login(code: str = Form("")):
    """Form login kept for old bookmarks/scripts; redirects into the app."""
    ok = hmac.compare_digest(code.strip(), ADMIN_CODE)
    return _with_session(RedirectResponse("/#overview" if ok else "/#qa", status_code=303), ok)


@router.get("/admin/logout")
def admin_logout():
    return _with_session(RedirectResponse("/#qa", status_code=303), False)


# Mount the protected API routes (must run after they are all defined).
router.include_router(_api)
