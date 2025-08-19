from __future__ import annotations
import os, uuid, shutil, tempfile
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from .config import settings
from .rag import answer_query, retrieve, get_chroma, ingest_folder
from .ingest import rebuild_from_data

app = FastAPI(title='Ezzogenics KB', version='0.1.4')

# CORS (open for prototype; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[orig.strip() for orig in settings.CORS_ORIGINS.split(',')],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Static
app.mount('/ui', StaticFiles(directory='public', html=True), name='ui')
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
def root():
    return FileResponse(os.path.join('public', 'index.html'))

@app.on_event("startup")
async def _log_startup():
    prov = (settings.LLM_PROVIDER or "").lower()
    model = settings.OPENAI_MODEL if prov == "openai" else settings.GROQ_MODEL
    key = settings.OPENAI_API_KEY if prov == "openai" else settings.GROQ_API_KEY
    key_prefix = (key[:10] + "…") if key else "(none)"
    print(f"[startup] provider={prov} model={model} key_prefix={key_prefix}")
    os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

@app.get('/health')
def health():
    return {'ok': True}

# -------------------- Ingestion --------------------

@app.post('/ingest/rebuild')
def ingest_rebuild():
    return rebuild_from_data()

# In-memory job store (fine for one process; if you scale, use Redis/DB)
_ingest_jobs: Dict[str, Dict[str, Any]] = {}

def _ingest_job(job_id: str, job_dir: str):
    """Run ingestion from a temp directory; update job status."""
    try:
        j = _ingest_jobs[job_id]
        j['status'] = 'processing'
        j['note'] = 'Parsing & embedding documents…'
        res = ingest_folder(job_dir)  # your rag.ingest_folder(data_dir)
        j['status'] = 'done'
        j['result'] = res
    except Exception as e:
        _ingest_jobs[job_id]['status'] = 'error'
        _ingest_jobs[job_id]['error'] = str(e)
    finally:
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass

@app.post('/ingest/upload')
async def ingest_upload(background: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Save uploads to a temp directory and kick off ingestion in the background.
    Returns 202 + job_id immediately to avoid Render's request timeouts.
    """
    if not files:
        return JSONResponse({'error': 'No files'}, status_code=400)

    job_id = str(uuid.uuid4())
    job_dir = tempfile.mkdtemp(prefix=f"ingest_{job_id}_")
    _ingest_jobs[job_id] = {
        'status': 'queued',
        'uploaded': [f.filename for f in files],
        'note': 'Waiting to start…',
    }

    # Persist uploads for the background task
    for f in files:
        dest = os.path.join(job_dir, f.filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

    background.add_task(_ingest_job, job_id, job_dir)
    return JSONResponse({'job_id': job_id, 'status': 'queued'}, status_code=202)

@app.get('/ingest/status/{job_id}')
def ingest_status(job_id: str):
    job = _ingest_jobs.get(job_id)
    if not job:
        return JSONResponse({'error': 'unknown job_id'}, status_code=404)
    return job

# -------------------- Chat --------------------

@app.post('/chat')
async def chat(payload: Dict[str, Any]):
    q = (payload or {}).get('query', '').strip()
    if not q:
        return JSONResponse({'error': 'Missing query'}, status_code=400)
    try:
        return answer_query(q)
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

# -------------------- Debug --------------------

@app.post('/debug/retrieve')
async def debug_retrieve(payload: Dict[str, Any]):
    q = (payload or {}).get('query', '').strip()
    k = int((payload or {}).get('k', settings.TOP_K))
    try:
        items = retrieve(q, k)
        return {'query': q, 'k': k, 'results': items}
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get('/debug/sources')
def debug_sources():
    col = get_chroma()
    res = col.get(include=['metadatas'], limit=100000)
    sources = sorted({m.get('source') for m in (res.get('metadatas') or []) if m and m.get('source')})
    return {'total_sources': len(sources), 'sources': sources}

@app.get('/sources')
def sources():
    return {'message': 'Ask via /chat; citations include file and page.'}
