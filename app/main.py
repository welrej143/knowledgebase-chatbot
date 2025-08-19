from __future__ import annotations

import os
import tempfile
import threading
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .ingest import rebuild_from_data, save_and_ingest_uploads
from .rag import answer_query, get_chroma, retrieve

app = FastAPI(title="Ezzogenics KB", version="0.2.0")

# CORS (open for prototype; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[orig.strip() for orig in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the chat UI at /
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
    print(
        f"[startup] provider={prov} model={model} key_prefix={(key[:10]+'…') if key else '(none)'}"
    )
    os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@app.get("/health")
def health():
    return {"ok": True}


# -------- Ingestion (rebuild) --------
@app.post("/ingest/rebuild")
def ingest_rebuild():
    return rebuild_from_data()


# ========= Non-blocking upload & ingest =========
_ingest_jobs: Dict[str, Dict[str, Any]] = {}


def _ingest_job(job_id: str, job_dir: str):
    """Run ingestion work in a background thread."""
    try:
        _ingest_jobs[job_id]["status"] = "processing"
        _ingest_jobs[job_id]["note"] = "Parsing and indexing…"

        # Build UploadFile objects from saved files
        to_close: List[Any] = []
        files: List[UploadFile] = []
        for name in os.listdir(job_dir):
            path = os.path.join(job_dir, name)
            if not os.path.isfile(path):
                continue
            fobj = open(path, "rb")
            to_close.append(fobj)
            files.append(UploadFile(filename=name, file=fobj))

        # Ingest
        res = save_and_ingest_uploads(files)

        # Clean up file handles & temp dir
        for fobj in to_close:
            try:
                fobj.close()
            except Exception:
                pass
        try:
            for name in os.listdir(job_dir):
                try:
                    os.remove(os.path.join(job_dir, name))
                except Exception:
                    pass
            os.rmdir(job_dir)
        except Exception:
            pass

        _ingest_jobs[job_id]["status"] = "done"
        _ingest_jobs[job_id]["result"] = res
        _ingest_jobs[job_id]["note"] = "Completed."
    except Exception as e:
        _ingest_jobs[job_id]["status"] = "error"
        _ingest_jobs[job_id]["error"] = str(e)
        _ingest_jobs[job_id]["note"] = "Failed during ingestion."


@app.post("/ingest/upload")
async def ingest_upload(files: List[UploadFile] = File(...)):
    """Accept files immediately, save to a temp dir, kick off a background thread, return 202."""
    if not files:
        return JSONResponse({"error": "No files provided"}, status_code=400)

    job_id = str(uuid.uuid4())
    job_dir = tempfile.mkdtemp(prefix=f"ingest_{job_id}_")

    # Save streams to disk (so the background thread can reopen them)
    uploaded_names: List[str] = []
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

    # Record job & start worker thread
    _ingest_jobs[job_id] = {
        "status": "queued",
        "uploaded": uploaded_names,
        "note": "Queued. Will start shortly…",
    }
    threading.Thread(target=_ingest_job, args=(job_id, job_dir), daemon=True).start()

    return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    job = _ingest_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job_id"}, status_code=404)
    return job


# ========= Chat & debug =========
@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    q = (payload or {}).get("query", "").strip()
    if not q:
        return JSONResponse({"error": "Missing query"}, status_code=400)
    try:
        return answer_query(q)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/debug/retrieve")
async def debug_retrieve(payload: Dict[str, Any]):
    q = (payload or {}).get("query", "").strip()
    k = int((payload or {}).get("k", settings.TOP_K))
    try:
        items = retrieve(q, k)
        return {"query": q, "k": k, "results": items}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/debug/sources")
def debug_sources():
    col = get_chroma()
    res = col.get(include=["metadatas"], limit=100000)
    sources = sorted(
        {
            m.get("source")
            for m in (res.get("metadatas") or [])
            if m and m.get("source")
        }
    )
    return {"total_sources": len(sources), "sources": sources}


@app.get("/sources")
def sources():
    return {"message": "Ask via /chat; citations include file and page."}
