import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from backend.config import settings
from backend.services.document_loader import load_pdf
from backend.services.chunker import chunk_pages
from backend.services.vector_store import add_documents_to_store, get_vectorstore_status
from backend.utils.models import UploadResponse, StatusResponse
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB = 50


def _apply_runtime_settings(request: Request) -> None:
    provider = request.headers.get("x-llm-provider", "").strip().lower()
    if provider in {"gemini", "openai"}:
        settings.llm_provider = provider

    openai_key = request.headers.get("x-openai-api-key", "").strip()
    if openai_key:
        settings.openai_api_key = openai_key

    gemini_key = request.headers.get("x-gemini-api-key", "").strip()
    if gemini_key:
        settings.gemini_api_key = gemini_key


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF, extract text, chunk it, and index into FAISS.
    """
    # Validate extension
    _apply_runtime_settings(request)

    filename = file.filename or "upload.pdf"
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only PDF files are supported. Got: {ext}")

    # Read bytes
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB.",
        )

    # Save to temp file and process
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pages = load_pdf(tmp_path)
        # Override source name with original filename
        for p in pages:
            p.source = filename

        chunks = chunk_pages(pages)
        add_documents_to_store(chunks)

        logger.info(f"Indexed '{filename}': {len(pages)} pages, {len(chunks)} chunks.")
        return UploadResponse(
            message="Document successfully processed and indexed.",
            filename=filename,
            pages_processed=len(pages),
            chunks_created=len(chunks),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed for '{filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request):
    """Return vectorstore health and document count."""
    _apply_runtime_settings(request)

    status = get_vectorstore_status()
    return StatusResponse(
        vectorstore_loaded=status["loaded"],
        document_count=status["document_count"],
    )
