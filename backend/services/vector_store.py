import os
from pathlib import Path
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cache so the index lives in memory between requests
_vectorstore: Optional[FAISS] = None


def _ensure_openai_key() -> None:
    key = (settings.openai_api_key or "").strip()
    if not key or "your-openai-api-key-here" in key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. Set a real key in .env and restart the backend."
        )


def _ensure_gemini_key() -> None:
    key = (settings.gemini_api_key or "").strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. Set a real key in .env and restart the backend."
        )


def _get_embeddings() -> Any:
    provider = (settings.llm_provider or "gemini").strip().lower()

    if provider == "openai":
        _ensure_openai_key()
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    if provider == "gemini":
        _ensure_gemini_key()
        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini' or 'openai'."
    )


def build_vectorstore(documents: List[Document]) -> FAISS:
    """
    Build a FAISS index from documents and persist it to disk.
    Replaces any existing index.
    """
    global _vectorstore

    if not documents:
        raise ValueError("Cannot build vectorstore from empty document list.")

    logger.info(f"Building FAISS index from {len(documents)} chunks...")

    embeddings = _get_embeddings()
    store = FAISS.from_documents(documents, embeddings)

    # Persist
    store_path = Path(settings.vectorstore_path)
    store_path.mkdir(parents=True, exist_ok=True)
    try:
        store.save_local(str(store_path))
    except PermissionError as exc:
        raise ValueError(
            f"Vectorstore path is not writable: {store_path}. Close other processes using it and retry."
        ) from exc

    _vectorstore = store
    logger.info(f"FAISS index saved to '{store_path}'.")
    return store


def load_vectorstore() -> Optional[FAISS]:
    """
    Load FAISS index from disk into memory cache.
    Returns None if no index exists yet.
    """
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    store_path = settings.vectorstore_path
    index_file = Path(store_path) / "index.faiss"

    if not index_file.exists():
        logger.warning("No FAISS index found on disk.")
        return None

    embeddings = _get_embeddings()
    _vectorstore = FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded from disk.")
    return _vectorstore


def add_documents_to_store(documents: List[Document]) -> FAISS:
    """
    Add new documents to existing index, or create if none exists.
    """
    global _vectorstore

    existing = load_vectorstore()
    if existing is None:
        return build_vectorstore(documents)

    logger.info(f"Adding {len(documents)} chunks to existing FAISS index...")
    embeddings = _get_embeddings()
    existing.add_documents(documents)
    try:
        existing.save_local(settings.vectorstore_path)
    except PermissionError as exc:
        raise ValueError(
            f"Vectorstore path is not writable: {settings.vectorstore_path}. Close other processes using it and retry."
        ) from exc
    _vectorstore = existing
    return existing


def get_vectorstore_status() -> dict:
    store = load_vectorstore()
    if store is None:
        return {"loaded": False, "document_count": 0}
    return {"loaded": True, "document_count": store.index.ntotal}
