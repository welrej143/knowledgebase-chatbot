from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any
import os

from .config import settings
from .rag import answer_query, retrieve, get_chroma
from .ingest import rebuild_from_data, save_and_ingest_uploads

app = FastAPI(title='Ezzogenics KB', version='0.1.3')

# CORS (open for prototype; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[orig.strip() for orig in settings.CORS_ORIGINS.split(',')],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Serve the UI from /ui; also serve index at /
app.mount('/ui', StaticFiles(directory='public', html=True), name='ui')

@app.get('/')
def root():
    index_path = os.path.join('public', 'index.html')
    return FileResponse(index_path)

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

# -------- Ingestion --------

@app.post('/ingest/rebuild')
def ingest_rebuild():
    return rebuild_from_data()

@app.post('/ingest/upload')
async def ingest_upload(files: List[UploadFile] = File(...)):
    return save_and_ingest_uploads(files)

# -------- Chat --------

@app.post('/chat')
async def chat(payload: Dict[str, Any]):
    q = (payload or {}).get('query', '').strip()
    if not q:
        return JSONResponse({'error': 'Missing query'}, status_code=400)
    try:
        return answer_query(q)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

# -------- Debug --------

@app.post('/debug/retrieve')
async def debug_retrieve(payload: Dict[str, Any]):
    q = (payload or {}).get('query', '').strip()
    k = int((payload or {}).get('k', settings.TOP_K))
    try:
        items = retrieve(q, k)
        return {'query': q, 'k': k, 'results': items}
    except Exception as e:
        import traceback
        traceback.print_exc()
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

# Serve static files (upload page)
app.mount("/static", StaticFiles(directory="static"), name="static")
