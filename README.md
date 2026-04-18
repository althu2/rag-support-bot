# 🤖 RAG Customer Support Chatbot

Production-ready AI chatbot using Retrieval-Augmented Generation (RAG).
Upload PDFs → Ask questions → Get answers with source citations.

---

## 📁 Project Structure

```
rag-support-bot/
├── main.py                          # FastAPI entry point
├── requirements.txt
├── .env                             # API keys (edit this)
│
├── backend/
│   ├── config.py                    # Settings via pydantic-settings
│   ├── routes/
│   │   ├── documents.py             # POST /documents/upload, GET /documents/status
│   │   └── chat.py                  # POST /chat/
│   ├── services/
│   │   ├── document_loader.py       # PyMuPDF PDF parser
│   │   ├── chunker.py               # RecursiveCharacterTextSplitter
│   │   ├── vector_store.py          # FAISS index (build/load/add)
│   │   └── rag_chain.py             # LCEL RAG chain (retrieve → prompt → LLM)
│   └── utils/
│       ├── logger.py                # Structured logging
│       └── models.py                # Pydantic request/response models
│
├── frontend/
│   ├── app.py                       # Streamlit chat UI
│   └── .streamlit/
│       └── config.toml              # Dark theme config
│
└── data/
    └── vectorstore/                 # FAISS index stored here (auto-created)
```

---

## ⚙️ Setup & Run (Step-by-Step)

### Step 1 — Prerequisites

- Python 3.10 or 3.11 installed
- An OpenAI API key (get one at https://platform.openai.com/api-keys)

---

### Step 2 — Clone / Open in VS Code

Open the `rag-support-bot/` folder in VS Code.

---

### Step 3 — Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

> This installs FastAPI, LangChain, OpenAI SDK, FAISS, PyMuPDF, Streamlit, and all dependencies.

---

### Step 5 — Configure Environment

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-key-here
```

All other values have sensible defaults and don't need to change.

---

### Step 6 — Start the FastAPI Backend

Open **Terminal 1** in VS Code:

```bash
# Make sure venv is activated
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO  | Starting RAG Customer Support API...
INFO  | No existing vectorstore found. Upload a document to get started.
INFO  | Uvicorn running on http://0.0.0.0:8000
```

API docs available at: http://localhost:8000/docs

---

### Step 7 — Start the Streamlit Frontend

Open **Terminal 2** in VS Code:

```bash
# Make sure venv is activated
streamlit run frontend/app.py
```

Browser opens automatically at: http://localhost:8501

---

### Step 8 — Use the Chatbot

1. The sidebar shows **API Online** (green dot)
2. Click **Browse files** → select a PDF
3. Click **📤 Process & Index** → wait for success message
4. Type a question in the input box → press Enter or click ➤
5. Answer appears with source citations (filename + page number)

---

## 🧪 Sample Test Case

### Test PDF
Use any PDF — for example, download a product manual or a company FAQ.

A good free test document:
- Go to https://www.irs.gov/pub/irs-pdf/p17.pdf (IRS Publication 17 — tax guide)
- Or use any company's terms of service PDF

### Expected Behavior

**Upload:** `company_faq.pdf` (10 pages)
→ "✅ Pages: 10 | Chunks: 47"

**Question:** "What is your return policy?"
→ Answer citing the relevant section with: `📄 company_faq.pdf · p.3`

**Question:** "What are your support hours?"
→ Answer with page reference

**Question:** "What is the capital of France?"
→ "I'm sorry, I don't have information about that in the uploaded documents."

**Question asked before any upload:**
→ "No documents have been uploaded yet. Please upload a PDF first."

---

## 🔧 Configuration Reference (.env)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | Your OpenAI key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model (cheapest/fastest) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |
| `VECTORSTORE_PATH` | `./data/vectorstore` | Where FAISS index is saved |

---

## 💡 Switching to GPT-4o (better quality)

In `.env`:
```env
OPENAI_CHAT_MODEL=gpt-4o
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv active |
| `API Offline` in UI | Start FastAPI with `uvicorn main:app --reload` |
| `OPENAI_API_KEY` error | Check `.env` has correct key, no quotes around it |
| Empty PDF error | PDF may be image-scanned — needs OCR (not supported in this version) |
| Port 8000 in use | Change port: `uvicorn main:app --port 8001` + update `API_BASE` in `frontend/app.py` |

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/documents/upload` | Upload + index a PDF |
| `GET` | `/documents/status` | Vectorstore status |
| `POST` | `/chat/` | Ask a question |

Full interactive docs: http://localhost:8000/docs
