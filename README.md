# Ezzogenics RAG Prototype (FastAPI + Chroma + SBERT + OpenAI/Groq)

A lightweight, *production-friendly* knowledge base with Retrieval-Augmented Generation (RAG).
- **Knowledge base**: PDFs → chunked → embedded → stored in **ChromaDB** (local persisted folder).
- **AI agent**: Answers questions grounded on your PDFs, with **citations**.
- **LLM**: Works with **Groq** (free-tier, Llama 3.1) or **OpenAI**. Pick via `.env` (`LLM_PROVIDER`).
- **Embeddings**: **all-MiniLM-L6-v2** (Sentence Transformers).
- **API**: FastAPI endpoints for ingestion, chat, and listing sources.
- **UI**: Minimal web chat at `/` (single HTML file).

> ✅ This meets the test goals: create a KB, parse PDFs into it, expose an AI chatbot that reads from it, and answer with citations. It’s easy to link later to a CRM via API (step 3).

---

## 1) Quick Start

### A. Local (Mac/Windows/Linux)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Open .env and set GROQ_API_KEY or OPENAI_API_KEY

# Put your PDFs in the ./data folder
python -m app.ingest

# Run the API + UI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### B. Replit
1. Create a new Repl (Python).
2. Upload **all files** from this repo (or the provided ZIP), including `requirements.txt`.
3. In **Secrets**, add `GROQ_API_KEY` (or `OPENAI_API_KEY`) and any other values from `.env`.
4. In Shell:
   ```bash
   pip install -r requirements.txt
   python -m app.ingest
   ```
5. Click **Run** or run:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
6. Open the webview; you’ll see the chat UI.

> Tip: You can also add a **Procfile** and use platform run buttons (already included).

### C. Docker (optional)
```bash
docker build -t ezzogenics-rag .
docker run -p 8000:8000 --env-file .env -v $(pwd)/storage:/app/storage -v $(pwd)/data:/app/data ezzogenics-rag
```

---

## 2) Endpoints

- `POST /ingest/upload` (multipart) → Upload PDFs now and ingest on the fly.
- `POST /ingest/rebuild` → Rebuild the vector store from the `./data` folder.
- `POST /chat` → Body: `{ "query": "your question" }` → returns AI answer + citations.
- `GET /sources` → Lists document names/pages present in the knowledge base.
- `GET /health` → Health check.

---

## 3) How It Works

1. **Ingestion** (`app/ingest.py`)
   - Reads PDFs from `./data` or from uploaded files.
   - Splits into ~800 token-like chunks (with overlap) at paragraph/sentence boundaries.
   - Embeds chunks with **Sentence Transformers** (`all-MiniLM-L6-v2`).
   - Persists embeddings to **ChromaDB** at `CHROMA_DIR` (default `./storage/chroma`).

2. **Retrieval + Generation** (`app/rag.py`)
   - Retrieves **TOP_K** most relevant chunks.
   - Builds a grounded prompt containing only retrieved context.
   - Calls **Groq** or **OpenAI** (your choice) to generate an answer.
   - Returns answer + **citations** listing each source file and page.

3. **UI** (`public/index.html`)
   - Minimal chat interface hitting `POST /chat`.
   - Renders citations under each answer.

---

## 4) Linking to a CRM (Step 3 of the test)

Later, you or the CRM vendor can connect to this KB by calling the REST API:
- Push new docs via `POST /ingest/upload`
- Trigger rebuilds via `POST /ingest/rebuild`
- Ask questions via `POST /chat`
- Track provenance via `GET /sources`

You can expose a **token-protected** version easily (add an API key header check in `app/main.py`).

---

## 5) Testing & Acceptance

After you ingest the PDFs provided by Ezzogenics:
1. Go to `/` and ask factual questions that your PDFs can answer.
2. Verify the response contains **correct citations** (file names + page numbers).
3. Try queries with domain terms, abbreviations, and synonyms.

Success criteria:
- Answers **grounded** in the PDFs (no hallucinations).
- Citations are present and relevant.
- Performance is acceptable (embedding size kept small deliberately).

---

## 6) Troubleshooting

- **Model downloads are slow**: first-time SBERT download can take a minute. It’s cached afterward.
- **Empty answers**: make sure you ran `python -m app.ingest` after placing PDFs in `./data`.
- **Change TOP_K**: in `.env` set `TOP_K=6` if your docs are broad.
- **Use OpenAI**: set `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`, `OPENAI_MODEL=gpt-4o-mini`.
- **Free-tier**: Set `LLM_PROVIDER=groq` and use `llama-3.1-70b-versatile` for strong results.

---

## 7) Security Notes

This is a prototype. Before production:
- Add **auth** (API key or OAuth) to all write endpoints (`/ingest/*`).
- Add **rate limits** and request validation.
- Consider **PII scrubbing** or redaction if sensitive content will be ingested.
- Consider **Doc versioning** and **role-based access** per source.

---

## 8) File/Folder Overview

```
app/
  main.py        # FastAPI app & endpoints
  rag.py         # Embedding, retrieval, generation
  ingest.py      # CLI/API ingestion
  config.py      # Settings via .env
public/
  index.html     # Simple chat UI
data/            # Put PDFs here (for /ingest/rebuild) or upload to /ingest/upload
storage/         # ChromaDB persistence (auto-created)
.env.example
requirements.txt
Procfile
```

---

## 9) Optional: Why This Stack vs NotebookLM / RAGFlow

- **Own stack** gives you *full control*, API endpoints, and portability (can run in your VPC).
- **RAGFlow/NotebookLM** are great too; if you prefer hosted:
  - Upload PDFs in their UI, connect your LLM key, and you already have a working chat.
  - To link to a CRM, ensure the platform exposes ext. APIs or webhooks. If not, keep this repo as your “middleware layer” that the CRM calls.
