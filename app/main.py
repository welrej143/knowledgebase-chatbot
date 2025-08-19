from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback as tb
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .ingest import rebuild_from_data, save_and_ingest_uploads
from .rag import answer_query, get_chroma, retrieve


app = FastAPI(title="Ezzogenics KB", version="0.2.2")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[orig.strip() for orig in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static/UI
app.mount("/ui", StaticFiles(directory="public", html=True), name="ui")
app.mount("/static", StaticFiles(directory="static", html=False), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join("public", "index.html"))


@app.on_event("startup")
async def _log_startup():
    prov = (settings.LLM_PROVIDER or "").lower()
    model = settings.OPENAI_MODEL if prov == "openai" else settings.GROQ_MODEL
    key = settings.OPENAI_API_KEY if prov == "openai" else settings.GROQ_API_KEY
    print(f"[startup] provider={prov} model={model} key_prefix={(key[:10]+'…') if key else '(none)'}")
    os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@app.get("/health")
def health():
    return {"ok": True}


# -------------------- Job store (disk-persisted) --------------------

JOBS_DIR = os.path.join("storage", "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")

def _job_save(job_id: str, obj: Dict[str, Any]) -> None:
    obj["updated_at"] = int(time.time())
    path = _job_path(job_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _job_load(job_id: str) -> Dict[str, Any] | None:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------- Rebuild (optional) --------------------

@app.post("/ingest/rebuild")
def ingest_rebuild():
    return rebuild_from_data()


# -------------------- Background ingestion --------------------

WATCHDOG_IDLE_SECS = 30 * 60  # 30 minutes

def _ingest_job(job_id: str, job_dir: str, uploaded: List[str]) -> None:
    """Run ingestion work in a background thread (status persisted to disk)."""
    try:
        _job_save(job_id, {
            "status": "processing",
            "uploaded": uploaded,
            "started_at": int(time.time()),
            "step": "opening-files",
            "note": "Parsing and indexing…"
        })

        # Reopen saved uploads for ingestion
        to_close: List[Any] = []
        files: List[UploadFile] = []
        for name in os.listdir(job_dir):
            p = os.path.join(job_dir, name)
            if not os.path.isfile(p):
                continue
            fobj = open(p, "rb")
            to_close.append(fobj)
            files.append(UploadFile(filename=name, file=fobj))

        _job_save(job_id, {
            "status": "processing",
            "uploaded": uploaded,
            "started_at": _job_load(job_id)["started_at"],
            "step": "ingest-start",
            "note": "Calling save_and_ingest_uploads…"
        })

        # Do the actual ingest
        res = save_and_ingest_uploads(files)

        # Cleanup temp files
        for fobj in to_close:
            try: fobj.close()
            except Exception: pass
        try:
            for name in os.listdir(job_dir):
                try: os.remove(os.path.join(job_dir, name))
                except Exception: pass
            os.rmdir(job_dir)
        except Exception:
            pass

        _job_save(job_id, {
            "status": "done",
            "uploaded": uploaded,
            "result": res,
            "started_at": _job_load(job_id)["started_at"],
            "step": "complete",
            "note": "Completed."
        })
    except Exception as e:
        _job_save(job_id, {
            "status": "error",
            "uploaded": uploaded,
            "error": str(e),
            "traceback": tb.format_exc(),
            "started_at": _job_load(job_id)["started_at"] if _job_load(job_id) else int(time.time()),
            "step": "exception",
            "note": "Failed during ingestion."
        })


@app.post("/ingest/upload")
async def ingest_upload(files: List[UploadFile] = File(...)):
    """Accept files, persist to temp dir, kick off a thread, return 202 with job_id."""
    if not files:
        return JSONResponse({"error": "No files provided"}, status_code=400)

    job_id = str(uuid.uuid4())
    job_dir = tempfile.mkdtemp(prefix=f"ingest_{job_id}_")
    uploaded_names: List[str] = []

    # Save the uploaded streams to disk for the worker
    for uf in files:
        dest = os.path.join(job_dir, uf.filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as out:
            while True:
                chunk = await uf.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        uploaded_names.append(uf.filename)

    _job_save(job_id, {
        "status": "queued",
        "uploaded": uploaded_names,
        "started_at": int(time.time()),
        "step": "queued",
        "note": "Queued. Will start shortly…"
    })

    threading.Thread(target=_ingest_job, args=(job_id, job_dir, uploaded_names), daemon=True).start()
    return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    job = _job_load(job_id)
    if not job:
        return JSONResponse({"error": "unknown job_id"}, status_code=404)

    # Watchdog: if processing and stale, mark as error to avoid “stuck”
    if job.get("status") in {"queued", "processing"}:
        updated = job.get("updated_at") or job.get("started_at") or int(time.time())
        if int(time.time()) - int(updated) > WATCHDOG_IDLE_SECS:
            job["status"] = "error"
            job["note"] = f"Watchdog timeout (> {WATCHDOG_IDLE_SECS//60} min without progress)."
            _job_save(job_id, job)

    return job


# -------------------- Chat & debug --------------------

@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    q = (payload or {}).get("query", "").strip()
    if not q:
        return JSONResponse({"error": "Missing query"}, status_code=400)
    try:
        return answer_query(q)
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/debug/retrieve")
async def debug_retrieve(payload: Dict[str, Any]):
    q = (payload or {}).get("query", "").strip()
    k = int((payload or {}).get("k", settings.TOP_K))
    try:
        items = retrieve(q, k)
        return {"query": q, "k": k, "results": items}
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/debug/sources")
def debug_sources():
    col = get_chroma()
    res = col.get(include=["metadatas"], limit=100000)
    sources = sorted({m.get("source") for m in (res.get("metadatas") or []) if m and m.get("source")})
    return {"total_sources": len(sources), "sources": sources}


@app.get("/sources")
def sources():
    return {"message": "Ask via /chat; citations include file and page."}
