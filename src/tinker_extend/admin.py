"""Admin console: visibility into Phase 0 (eval) and Phase 1 (ingestion).

parallel.ai-inspired developer console: stage cards, live run feed,
benchmark-style fidelity tables, and a dead-letter queue — everything
needed to see what is actually happening in the pipeline.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import threading
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .config import load_config
from .embeddings import embedder_from_env
from .extend_client import ExtendClient
from .fidelity import check_golden_doc, fidelity_metrics, load_golden_docs, red_flags
from .ingest import SUPPORTED_EXTENSIONS, iter_candidate_files, sha256_file
from .pipeline import WaveIngester
from .store import KnowledgeBase

# Page routes (/admin, login, logout) are public; every /api/admin route is
# registered on _api, which requires the admin code. _api is mounted onto
# router at the bottom of this module, after all its routes exist.
router = APIRouter()

KB_PATH = os.environ.get("TINKER_KB_PATH", "./kb.sqlite3")
SEED_DIR = os.environ.get("TINKER_SEED_DIR", "./seed_docs")
UPLOAD_DIR = os.environ.get("TINKER_UPLOAD_DIR", "./uploads")
MAX_UPLOAD_BYTES = 150 * 1024 * 1024  # per file

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


def _run_wave_job(wave: str, paths: list[str], workers: int) -> None:
    with _JOBS_LOCK:
        JOBS[wave] = {"status": "running", "progress": {"total": len(paths)},
                      "started_at": time.time()}
    try:
        ingester = WaveIngester(_client_factory, KB_PATH, workers=workers,
                                embedder=embedder_from_env())

        def on_progress(progress: dict, res: dict) -> None:
            with _JOBS_LOCK:
                JOBS[wave]["progress"] = progress
                JOBS[wave]["last"] = {"source": os.path.basename(res["source"]),
                                     "status": res["status"]}

        summary = ingester.run(paths, wave, progress_cb=on_progress)
        summary.pop("results", None)
        with _JOBS_LOCK:
            JOBS[wave] = {"status": "done", "progress": summary,
                          "started_at": JOBS[wave]["started_at"]}
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
                             "doc_type": r["doc_type"], "flags": fl})
        return {"by_type": by_type, "worst": worst[:25]}
    finally:
        kb.close()


@_api.get("/api/admin/golden_docs")
def api_golden_docs():
    if not os.path.exists(GOLDEN_DOCS):
        return {"specs": [], "note": "no golden docs defined yet"}
    kb = _kb()
    try:
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
        return {"specs": results}
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


def recompute_doc_telemetry(kb: KnowledgeBase, doc_id: str) -> dict | None:
    """Recompute fidelity metrics from stored blocks (no API calls).

    Used when the metric code changes or a row was recorded with a bug.
    Wave/engine/cost/duration are preserved.
    """
    doc = kb.conn.execute(
        "SELECT doc_id, source FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
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
    src = doc["source"]
    if src and os.path.exists(src):
        try:
            from pypdf import PdfReader
            text = "\n".join((p.extract_text() or "") for p in PdfReader(src).pages)
            if text.strip():
                from .fidelity import text_coverage as _tc
                metrics["text_coverage"] = _tc(text, "\n".join(chunks))
        except Exception:
            pass
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


@router.get("/admin", response_class=HTMLResponse)
def admin_index(request: Request):
    if not is_admin(request):
        return HTMLResponse(_login_page())
    return ADMIN_PAGE


@router.post("/admin/login")
def admin_login(code: str = Form("")):
    if not hmac.compare_digest(code.strip(), ADMIN_CODE):
        return HTMLResponse(_login_page("Wrong code."), status_code=401)
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie(ADMIN_COOKIE, _session_token(), max_age=ADMIN_COOKIE_MAX_AGE,
                    httponly=True, samesite="lax")
    return resp


@router.get("/admin/logout")
def admin_logout():
    resp = RedirectResponse("/admin", status_code=303)
    resp.delete_cookie(ADMIN_COOKIE)
    return resp


def _login_page(error: str = "") -> str:
    err = f'<p class="err">{error}</p>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tinkerExtend Console</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
         font-family: -apple-system, "Segoe UI", Inter, sans-serif;
         background: #0a0c10; color: #e8ecf1; }}
  form {{ width: min(320px, calc(100vw - 32px)); background: #0d1015;
         border: 1px solid #1c222b; border-radius: 10px; padding: 24px; }}
  h1 {{ font-size: 18px; margin: 0 0 16px; }}
  input, button {{ width: 100%; box-sizing: border-box; padding: 10px;
                  border-radius: 6px; font-size: 14px; }}
  input {{ background: #0a0c10; color: #e8ecf1; border: 1px solid #2a323d;
          margin-bottom: 12px; }}
  button {{ background: #3b82f6; color: #fff; border: 0; cursor: pointer; }}
  .err {{ color: #f87171; font-size: 13px; margin: 0 0 12px; }}
</style>
</head>
<body>
<form method="post" action="/admin/login">
  <h1>tinkerExtend console</h1>
  {err}
  <input type="password" name="code" placeholder="Access code" autofocus required>
  <button type="submit">Enter</button>
</form>
</body>
</html>"""


ADMIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tinkerExtend Console</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", Inter, sans-serif;
         background: #0a0c10; color: #e8ecf1; display: flex; min-height: 100vh; }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  /* sidebar */
  nav { width: 228px; flex-shrink: 0; background: #0d1015; border-right: 1px solid #1c222b;
        padding: 20px 14px; position: sticky; top: 0; height: 100vh; display: flex; flex-direction: column; }
  nav .brand { font-weight: 700; font-size: 16px; margin: 0 6px 4px; }
  nav .brand span { color: #4f7cff; }
  nav .tag { font-size: 11px; color: #697384; margin: 0 6px 18px; }
  nav a.item { display: block; padding: 9px 12px; border-radius: 8px; color: #a7b0bf;
    text-decoration: none; font-size: 14px; margin-bottom: 2px; cursor: pointer; border: 1px solid transparent; }
  nav a.item:hover { background: #141922; color: #e8ecf1; }
  nav a.item.active { background: #16202f; color: #fff; border-color: #24344d; }
  nav .foot { margin-top: auto; font-size: 12px; color: #697384; padding: 0 6px; }
  nav .foot a { color: #7ea2ff; text-decoration: none; }
  /* main */
  main { flex: 1; padding: 22px 28px; max-width: 1280px; }
  .topbar { display: flex; align-items: center; gap: 12px; margin-bottom: 22px; }
  .topbar h1 { font-size: 20px; margin: 0; font-weight: 650; }
  .pill { font-size: 11px; padding: 3px 10px; border-radius: 20px; border: 1px solid #2a3340;
    color: #9aa5b5; background: #12161d; }
  .pill.live { color: #7ee2a0; border-color: #1f4d33; background: #0f1f16; }
  .topbar .sp { flex: 1; }
  button { font-family: inherit; }
  .btn { background: #2f6fed; color: #fff; border: 0; border-radius: 8px; padding: 9px 16px;
    font-size: 13.5px; font-weight: 600; cursor: pointer; }
  .btn:hover { background: #3b7bff; }
  .btn.ghost { background: #161c25; border: 1px solid #2a3340; color: #dbe2ec; }
  .btn.small { padding: 5px 11px; font-size: 12px; }
  .btn:disabled { opacity: .5; cursor: default; }
  /* stage cards (product-card pattern) */
  .stages { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 18px; }
  .stage { background: #10141b; border: 1px solid #1e242e; border-radius: 12px; padding: 16px 18px; }
  .stage .k { font-size: 11px; letter-spacing: .12em; color: #697384; font-weight: 700; margin-bottom: 8px; }
  .stage .v { font-size: 26px; font-weight: 700; }
  .stage .d { font-size: 12.5px; color: #8b94a3; margin-top: 6px; line-height: 1.45; }
  .stage .bar { height: 5px; background: #1c222b; border-radius: 3px; margin-top: 12px; overflow: hidden; }
  .stage .bar i { display: block; height: 100%; background: #2f6fed; border-radius: 3px; }
  /* kpi row */
  .kpis { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 18px; }
  .kpi { background: #10141b; border: 1px solid #1e242e; border-radius: 12px; padding: 14px 16px; }
  .kpi .k { font-size: 11px; color: #697384; margin-bottom: 6px; }
  .kpi .v { font-size: 21px; font-weight: 700; }
  .kpi .v.bad { color: #f85149; } .kpi .v.good { color: #3fb950; }
  /* panels */
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .panel { background: #10141b; border: 1px solid #1e242e; border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; }
  .panel h3 { margin: 0 0 12px; font-size: 14px; font-weight: 650; }
  .panel h3 .sub { color: #697384; font-weight: 400; font-size: 12px; margin-left: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-size: 11px; color: #697384; font-weight: 600; padding: 8px 10px;
       border-bottom: 1px solid #1e242e; white-space: nowrap; }
  td { padding: 9px 10px; border-bottom: 1px solid #161b23; vertical-align: top; }
  tr:last-child td { border-bottom: 0; }
  .num { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12.5px; }
  .status { font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 12px; white-space: nowrap; }
  .status.ok { color: #3fb950; background: #0f1f16; border: 1px solid #1f4d33; }
  .status.failed { color: #f85149; background: #230f0f; border: 1px solid #5c1e1e; }
  .status.running { color: #6ea8ff; background: #0f1a2e; border: 1px solid #274a7d; }
  .status.skipped { color: #8b94a3; background: #141922; border: 1px solid #2a3340; }
  .tag2 { font-size: 11px; padding: 2px 8px; border-radius: 6px; background: #161c25;
    border: 1px solid #2a3340; color: #a7b0bf; white-space: nowrap; }
  .tag2.perf { color: #d9a7ff; border-color: #4a2d6b; background: #1c1226; }
  .wbar { height: 6px; background: #1c222b; border-radius: 3px; overflow: hidden; min-width: 120px; }
  .wbar i { display: block; height: 100%; background: #2f6fed; }
  .wbar i.warn { background: #d29922; }
  .flag { background: #1a1210; border: 1px solid #4a2c1a; border-radius: 8px; padding: 10px 12px;
    margin-bottom: 8px; font-size: 13px; }
  .flag b { color: #f0b46a; }
  .flag .fl { color: #c98f5f; font-size: 12.5px; margin-top: 4px; }
  .empty { color: #697384; font-size: 13px; padding: 18px 0; text-align: center; }
  .form { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .form label { font-size: 12px; color: #8b94a3; display: block; margin-bottom: 5px; }
  .form input { width: 100%; background: #0d1117; border: 1px solid #2a3340; color: #e8ecf1;
    border-radius: 8px; padding: 9px 11px; font-size: 13.5px; font-family: inherit; }
  .gcard { background: #10141b; border: 1px solid #1e242e; border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; }
  .gcard .gh { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .gcard .gh b { font-size: 14px; }
  .probe { font-size: 13px; padding: 7px 0; border-bottom: 1px solid #161b23; display: flex; gap: 10px; }
  .probe:last-child { border: 0; }
  .probe .ok { color: #3fb950; font-weight: 700; } .probe .no { color: #f85149; font-weight: 700; }
  .probe .miss { color: #697384; font-size: 12px; }
  .tab { display: none; } .tab.active { display: block; }
  .err { color: #f85149; font-size: 12.5px; margin-top: 8px; }
  .okmsg { color: #3fb950; font-size: 12.5px; margin-top: 8px; }
  @media (max-width: 1000px) { .stages, .kpis { grid-template-columns: 1fr 1fr; } .grid2 { grid-template-columns: 1fr; } nav { display: none; } }
</style>
</head>
<body>
<nav>
  <div class="brand">tinker<span>Extend</span></div>
  <div class="tag">CONSOLE · PHASE 0/1</div>
  <a class="item active" data-tab="overview">Overview</a>
  <a class="item" data-tab="ingest">Ingestion</a>
  <a class="item" data-tab="fidelity">Parse Fidelity</a>
  <a class="item" data-tab="golden">Golden Sets</a>
  <a class="item" data-tab="dead">Dead Letter <span id="dlBadge"></span></a>
  <div class="foot"><a href="/">← Q&amp;A app</a> · <a href="/admin/logout">Sign out</a></div>
</nav>
<main>
  <div class="topbar">
    <h1 id="pageTitle">Overview</h1>
    <span class="pill live">● LIVE</span>
    <span class="sp"></span>
    <button class="btn ghost small" onclick="refresh()">Refresh</button>
  </div>

  <section class="tab active" id="tab-overview">
    <div class="stages" id="stages"></div>
    <div class="kpis" id="kpis"></div>
    <div class="grid2">
      <div class="panel"><h3>Waves <span class="sub">100 → 1k → 10k</span></h3><div id="waves"></div></div>
      <div class="panel"><h3>Red flags <span class="sub">parse fidelity</span></h3><div id="flags"></div></div>
    </div>
    <div class="panel"><h3>Recent activity</h3><div id="recent"></div></div>
  </section>

  <section class="tab" id="tab-ingest">
    <div class="panel">
      <h3>Upload documents <span class="sub">from your machine to the server</span></h3>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <input type="file" id="upFiles" multiple style="color:#a7b0bf;font-size:13px">
        <button class="btn" id="upBtn" onclick="uploadFiles()">Upload</button>
        <span class="num" style="font-size:12px;color:#697384" id="upStatus"></span>
      </div>
      <div class="err" id="upErr"></div>
      <div style="margin-top:10px;font-size:13px;color:#8b94a3" id="upList"></div>
    </div>
    <div class="panel">
      <h3>Start a wave</h3>
      <div class="form">
        <div><label>Source directory</label><input id="wDir" value="./seed_docs"></div>
        <div><label>Wave name</label><input id="wName" placeholder="pilot-100"></div>
        <div><label>File limit (0 = all)</label><input id="wLimit" type="number" value="100"></div>
        <div><label>Workers</label><input id="wWorkers" type="number" value="4"></div>
      </div>
      <div style="margin-top:12px"><button class="btn" id="wStart" onclick="startWave()">Start wave</button></div>
      <div class="err" id="wErr"></div><div class="okmsg" id="wOk"></div>
    </div>
    <div class="panel"><h3>Live jobs</h3><div id="jobs"></div></div>
    <div class="panel"><h3>Documents <span class="sub">per-document parse telemetry</span></h3><div id="docs"></div></div>
  </section>

  <section class="tab" id="tab-fidelity">
    <div class="panel"><h3>Fidelity by document type <span class="sub">benchmark view</span></h3><div id="byType"></div></div>
    <div class="panel"><h3>Needs review</h3><div id="worst"></div></div>
  </section>

  <section class="tab" id="tab-golden">
    <div class="panel"><h3>Golden documents <span class="sub">probe facts asserted every run</span></h3><div id="gdocs"></div></div>
    <div class="panel"><h3>Golden Q&amp;A <span class="sub">retrieval + answer baseline</span></h3><div id="gqa"></div></div>
  </section>

  <section class="tab" id="tab-dead">
    <div class="panel"><h3>Dead-letter queue <span class="sub">failures are never silently dropped</span></h3><div id="dl"></div></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = t => { const d = document.createElement('div'); d.textContent = t == null ? '' : String(t); return d.innerHTML; };
const pct = x => (x * 100).toFixed(1) + '%';
const money = x => '$' + Number(x || 0).toFixed(2);

document.querySelectorAll('nav a.item').forEach(a => {
  a.onclick = () => {
    document.querySelectorAll('nav a.item').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    a.classList.add('active');
    $('tab-' + a.dataset.tab).classList.add('active');
    $('pageTitle').textContent = a.textContent.trim();
    loadTab(a.dataset.tab);
  };
});

function statusPill(s){
  const c = s === 'ok' ? 'ok' : s === 'failed' ? 'failed' : s === 'running' ? 'running' : 'skipped';
  return `<span class="status ${c}">${esc(s).toUpperCase()}</span>`;
}
function engineBadge(e){
  const cls = e === 'parse_performance' ? 'tag2 perf' : 'tag2';
  const label = e === 'parse_performance' ? 'PERFORMANCE' : e === 'parse_light' ? 'LIGHT' : esc(e || '—');
  return `<span class="${cls}">${label}</span>`;
}
function waveBar(done, total){
  const p = total ? Math.round(done / total * 100) : 0;
  return `<div class="wbar"><i style="width:${p}%"></i></div><span class="num" style="font-size:12px">${done}/${total}</span>`;
}

async function api(path, opts){
  const r = await fetch(path, opts);
  if (r.status === 401) { location.href = '/admin'; throw new Error('signed out'); }
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

async function loadTab(t){
  if (t === 'overview') loadOverview();
  if (t === 'ingest') loadIngest();
  if (t === 'fidelity') loadFidelity();
  if (t === 'golden') loadGolden();
  if (t === 'dead') loadDead();
}
function refresh(){ loadTab(document.querySelector('nav a.item.active').dataset.tab); }

async function loadOverview(){
  const d = await api('/api/admin/overview');
  const o = d.overview;
  $('stages').innerHTML = `
    <div class="stage"><div class="k">INGEST</div><div class="v mono">${o.docs}</div>
      <div class="d">Documents parsed through Extend into the knowledge base, in measured waves.</div>
      <div class="bar"><i style="width:${Math.min(100, o.docs / 100 * 100)}%"></i></div></div>
    <div class="stage"><div class="k">PARSE</div><div class="v mono">${pct(o.success_rate)}</div>
      <div class="d">Parse success rate. Fidelity metrics computed per document at ingest.</div>
      <div class="bar"><i style="width:${o.success_rate * 100}%"></i></div></div>
    <div class="stage"><div class="k">EVALUATE</div><div class="v mono">${d.red_flags.length}</div>
      <div class="d">Documents currently raising fidelity red flags. Golden sets assert every run.</div>
      <div class="bar"><i class="${d.red_flags.length ? 'warn' : ''}" style="width:${Math.min(100, d.red_flags.length * 20)}%"></i></div></div>`;
  $('kpis').innerHTML = `
    <div class="kpi"><div class="k">DOCUMENTS</div><div class="v mono">${o.docs}</div></div>
    <div class="kpi"><div class="k">PARSE SUCCESS</div><div class="v mono ${o.success_rate < 0.95 ? 'bad' : 'good'}">${pct(o.success_rate)}</div></div>
    <div class="kpi"><div class="k">EST. SPEND</div><div class="v mono">${money(o.cost_usd)}</div></div>
    <div class="kpi"><div class="k">DEAD LETTERS</div><div class="v mono ${o.dead_letters ? 'bad' : ''}">${o.dead_letters}</div></div>
    <div class="kpi"><div class="k">FEEDBACK 👍</div><div class="v mono">${o.feedback_positive}/${o.feedback_total}</div></div>
    <div class="kpi" title="${esc(d.vectors.model || 'embeddings disabled')}"><div class="k">VECTORS</div><div class="v mono ${d.vectors.model && d.vectors.embedded < d.vectors.chunks ? 'bad' : ''}">${d.vectors.model ? d.vectors.embedded + '/' + d.vectors.chunks : 'off'}</div></div>`;
  $('dlBadge').textContent = o.dead_letters ? `(${o.dead_letters})` : '';
  $('waves').innerHTML = d.waves.length ? `<table><tr><th>Wave</th><th>Progress</th><th>OK</th><th>Failed</th><th>Cost</th><th>Status</th></tr>` +
    d.waves.map(w => `<tr><td><b>${esc(w.name)}</b></td><td><div style="display:flex;gap:8px;align-items:center">${waveBar(w.ingested, w.target)}</div></td>
      <td class="num">${w.ok}</td><td class="num">${w.failed}</td><td class="num">${money(w.cost_usd)}</td><td>${statusPill(w.status)}</td></tr>`).join('') + `</table>`
    : `<div class="empty">No waves yet. Start one from the Ingestion tab.</div>`;
  $('flags').innerHTML = d.red_flags.length ? d.red_flags.map(f =>
    `<div class="flag"><b>${esc(f.doc)}</b> <span class="tag2">${esc(f.doc_type || 'other')}</span>
     <div class="fl">${f.flags.map(esc).join(' · ')}</div></div>`).join('')
    : `<div class="empty">No red flags. Parse fidelity is clean.</div>`;
  $('recent').innerHTML = d.recent.length ? `<table><tr><th>Document</th><th>Type</th><th>Engine</th><th>Pages</th><th>Chunks</th><th>Tables</th><th>Empty</th><th>Cost</th><th>Status</th></tr>` +
    d.recent.map(r => `<tr><td>${esc(r.source.split('/').pop())}</td><td><span class="tag2">${esc(r.doc_type || '—')}</span></td>
      <td>${engineBadge(r.engine)}</td><td class="num">${r.num_pages}</td><td class="num">${r.num_chunks}</td>
      <td class="num">${r.num_tables}</td><td class="num">${pct(r.num_chunks ? r.empty_chunks / r.num_chunks : 0)}</td>
      <td class="num">${money(r.cost_usd)}</td><td>${statusPill(r.status)}</td></tr>`).join('') + `</table>`
    : `<div class="empty">Nothing ingested yet.</div>`;
}

async function loadIngest(){
  const [t, ov] = await Promise.all([api('/api/admin/telemetry?limit=100'), api('/api/admin/overview')]);
  listUploads();
  $('docs').innerHTML = t.rows.length ? `<table><tr><th>Document</th><th>Wave</th><th>Type</th><th>Engine</th><th>Pages</th><th>Chunks</th><th>Tables</th><th>OCR</th><th>Coverage</th><th>Cost</th><th>Time</th><th>Status</th><th>Flags</th></tr>` +
    t.rows.map(r => `<tr><td>${esc(r.name)}</td><td><span class="tag2">${esc(r.wave || '—')}</span></td>
      <td><span class="tag2">${esc(r.doc_type || '—')}</span></td><td>${engineBadge(r.engine)}</td>
      <td class="num">${r.num_pages}</td><td class="num">${r.num_chunks}</td><td class="num">${r.num_tables}</td>
      <td class="num">${r.ocr_confidence == null ? '—' : Number(r.ocr_confidence).toFixed(2)}</td>
      <td class="num">${r.text_coverage == null ? '—' : pct(r.text_coverage)}</td>
      <td class="num">${money(r.cost_usd)}</td><td class="num">${Number(r.duration_s || 0).toFixed(0)}s</td>
      <td>${statusPill(r.status)}</td>
      <td>${r.flags.length ? `<span style="color:#f0b46a;font-size:12px">${esc(r.flags[0])}${r.flags.length > 1 ? ` (+${r.flags.length - 1})` : ''}</span>` : '<span style="color:#3fb950">—</span>'}
      <br><button class="btn ghost small" style="margin-top:4px" onclick="recomputeRow('${r.doc_id}', this)">↻ recompute</button></td></tr>`).join('') + `</table>`
    : `<div class="empty">No telemetry yet.</div>`;
  renderJobs(ov.jobs);
  pollJobs();
}

async function recomputeRow(docId, btn){
  btn.disabled = true;
  try {
    await api('/api/admin/telemetry/recompute', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({doc_id: docId})});
    loadIngest();
  } catch (e) { alert('recompute failed: ' + e.message); btn.disabled = false; }
}

function renderJobs(jobs){
  const names = Object.keys(jobs);
  $('jobs').innerHTML = names.length ? names.map(n => {
    const j = jobs[n];
    const p = j.progress || {};
    const bar = p.total ? waveBar((p.done || 0) + (p.failed || 0) + (p.skipped || 0), p.total) : '';
    return `<div style="margin-bottom:10px"><div style="display:flex;gap:10px;align-items:center;margin-bottom:6px">
      <b>${esc(n)}</b>${statusPill(j.status === 'done' ? 'ok' : j.status === 'running' ? 'running' : 'failed')}
      <span style="display:flex;gap:8px;align-items:center">${bar}</span>
      ${j.last ? `<span style="font-size:12px;color:#697384">${esc(j.last.source)} → ${esc(j.last.status)}</span>` : ''}</div></div>`;
  }).join('') : `<div class="empty">No wave jobs running.</div>`;
}
let jobTimer = null;
async function pollJobs(){
  clearInterval(jobTimer);
  const tick = async () => {
    const d = await api('/api/admin/jobs');
    renderJobs(d.jobs);
    if (!Object.values(d.jobs).some(j => j.status === 'running')) { clearInterval(jobTimer); refresh(); }
  };
  jobTimer = setInterval(tick, 3000);
}

async function uploadFiles(){
  $('upErr').textContent = '';
  const input = $('upFiles');
  if (!input.files.length) { $('upErr').textContent = 'Choose files first.'; return; }
  const btn = $('upBtn'); btn.disabled = true;
  $('upStatus').textContent = 'uploading…';
  try {
    const fd = new FormData();
    for (const f of input.files) fd.append('files', f);
    const d = await api('/api/admin/upload', {method: 'POST', body: fd});
    $('upStatus').textContent = `${d.saved.length} uploaded${d.skipped.length ? `, ${d.skipped.length} skipped` : ''}`;
    if (d.skipped.length) $('upErr').textContent = d.skipped.map(s => `${s.file}: ${s.reason}`).join('; ');
    // Point the wave launcher at the upload dir so one click starts the wave.
    $('wDir').value = d.dir;
    if (!$('wName').value) $('wName').value = 'upload-' + new Date().toISOString().slice(0, 10);
    listUploads();
  } catch (e) { $('upErr').textContent = e.message; $('upStatus').textContent = ''; }
  btn.disabled = false;
}
async function listUploads(){
  try {
    const d = await api('/api/admin/uploads');
    $('upList').innerHTML = d.files.length
      ? `<b style="color:#dbe2ec">${d.files.length} file(s)</b> in <span class="mono" style="font-size:12px">${esc(d.dir)}</span><br>
         <span style="font-size:12px">${d.files.slice(0, 12).map(esc).join(', ')}${d.files.length > 12 ? ` (+${d.files.length - 12} more)` : ''}</span>`
      : 'No uploaded files yet.';
  } catch (e) { /* ignore */ }
}

async function startWave(){
  $('wErr').textContent = ''; $('wOk').textContent = '';
  const btn = $('wStart'); btn.disabled = true;
  try {
    const d = await api('/api/admin/waves', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({dir: $('wDir').value, wave: $('wName').value,
        limit: parseInt($('wLimit').value || '0', 10), workers: parseInt($('wWorkers').value || '4', 10)})});
    $('wOk').textContent = `Wave "${d.wave}" started (${d.files} files). Watch it below.`;
    pollJobs();
  } catch (e) { $('wErr').textContent = e.message; }
  btn.disabled = false;
}

async function loadFidelity(){
  const d = await api('/api/admin/fidelity');
  $('byType').innerHTML = d.by_type.length ? `<table><tr><th>Doc type</th><th>Docs</th><th>Table shape</th><th>Empty</th><th>Garbled</th><th>OCR conf</th><th>Coverage</th><th>Spend</th></tr>` +
    d.by_type.map(b => `<tr><td><span class="tag2">${esc(b.doc_type || 'other')}</span></td><td class="num">${b.docs}</td>
      <td class="num">${pct(b.table_shape_rate || 0)}</td><td class="num">${pct(b.empty_rate || 0)}</td>
      <td class="num">${pct(b.garbled_rate || 0)}</td><td class="num">${b.ocr_confidence == null ? '—' : Number(b.ocr_confidence).toFixed(3)}</td>
      <td class="num">${b.text_coverage == null ? '—' : pct(b.text_coverage)}</td><td class="num">${money(b.cost_usd)}</td></tr>`).join('') + `</table>`
    : `<div class="empty">No fidelity data yet — ingest a wave first.</div>`;
  $('worst').innerHTML = d.worst.length ? d.worst.map(f =>
    `<div class="flag"><b>${esc(f.doc)}</b> <span class="tag2">${esc(f.doc_type || 'other')}</span>
     <div class="fl">${f.flags.map(esc).join(' · ')}</div></div>`).join('')
    : `<div class="empty">Nothing needs review.</div>`;
}

async function loadGolden(){
  const [gd, gq] = await Promise.all([api('/api/admin/golden_docs'), api('/api/admin/golden_qa')]);
  $('gdocs').innerHTML = gd.specs.length ? gd.specs.map(s => `
    <div class="gcard"><div class="gh">
      <span class="status ${s.ok ? 'ok' : 'failed'}">${s.ok ? 'PASS' : s.status === 'not_ingested' ? 'SKIPPED' : 'FAIL'}</span>
      <b>${esc(s.doc)}</b><span class="tag2">${esc(s.doc_type || '')}</span>
      <span class="num" style="font-size:12px;color:#8b94a3">${s.probes_passed}/${s.probes_total} probes</span></div>
      ${s.probes.map(p => `<div class="probe"><span class="${p.passed ? 'ok' : 'no'}">${p.passed ? '✓' : '✗'}</span>
        <span>${esc(p.fact)}${p.missing.length ? ` <span class="miss">missing: ${p.missing.map(esc).join(', ')}</span>` : ''}</span></div>`).join('')}
      ${s.tables.length ? `<div style="font-size:12px;color:#697384;margin-top:8px">Tables: ${s.tables.map(t => esc(t.description) + ' (p.' + t.page + ')').join(' · ')}</div>` : ''}
    </div>`).join('')
    : `<div class="empty">${esc(gd.note || 'No golden docs.')}</div>`;
  const modeRow = (name, m) => m ? `<tr><td><b>${name}</b></td><td class="num">${m.hit_rate.toFixed(2)}</td>
    <td class="num">${m.mrr.toFixed(2)}</td><td class="num">${m.answer_mean.toFixed(2)}</td>
    <td class="num">${pct(m.abstention_rate || 0)}</td></tr>`
    : `<tr><td><b>${name}</b></td><td colspan="4" style="color:#697384">embeddings disabled</td></tr>`;
  const misses = ['keyword', 'hybrid'].filter(k => gq[k] && gq[k].misses.length)
    .map(k => `${k}: ${gq[k].misses.map(esc).join(' · ')}`);
  $('gqa').innerHTML = gq.cases ? `
    <div style="font-size:12px;color:#697384;margin-bottom:8px">${gq.cases} cases · k=${gq.k} · model ${esc(gq.model || 'none')}</div>
    <table><tr><th>Search</th><th>Hit rate@5</th><th>MRR</th><th>Answer mean</th><th>Abstention</th></tr>
    ${modeRow('Keyword', gq.keyword)}${modeRow('Hybrid', gq.hybrid)}</table>
    ${misses.length ? `<div style="margin-top:10px;font-size:13px;color:#f0b46a">Retrieval misses — ${misses.join(' | ')}</div>` : ''}`
    : `<div class="empty">${esc(gq.note || 'No golden Q&A.')}</div>`;
}

async function loadDead(){
  const d = await api('/api/admin/dead_letters');
  $('dl').innerHTML = d.rows.length ? `<table><tr><th>Source</th><th>Wave</th><th>Attempts</th><th>Error</th><th></th></tr>` +
    d.rows.map(r => `<tr><td class="mono" style="font-size:12px">${esc(r.source)}</td><td><span class="tag2">${esc(r.wave)}</span></td>
      <td class="num">${r.attempts}</td><td style="font-size:12px;color:#c98f5f;max-width:340px">${esc(r.last_error.slice(0, 160))}</td>
      <td style="white-space:nowrap"><button class="btn ghost small" onclick="dlRetry(${r.id})">Retry</button>
      <button class="btn ghost small" onclick="dlResolve(${r.id})">Resolve</button></td></tr>`).join('') + `</table>`
    : `<div class="empty">Queue is empty. Every failure lands here instead of vanishing.</div>`;
}
async function dlRetry(id){
  try { const d = await api(`/api/admin/dead_letters/${id}/retry`, {method: 'POST'});
    alert('Retrying as wave ' + d.wave); loadDead(); }
  catch (e) { alert(e.message); }
}
async function dlResolve(id){
  await api(`/api/admin/dead_letters/${id}/resolve`, {method: 'POST'});
  loadDead(); refresh();
}

loadOverview();
</script>
</body>
</html>
"""


# Mount the protected API routes (must run after they are all defined).
router.include_router(_api)
