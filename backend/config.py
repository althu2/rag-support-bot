import os
from pydantic_settings import BaseSettings
from pathlib import Path


def _sanitize_broken_proxy_env() -> None:
    proxy_vars = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]
    bad_markers = ("127.0.0.1:9", "localhost:9")
    for var in proxy_vars:
        value = (os.environ.get(var) or "").strip().lower()
        if any(marker in value for marker in bad_markers):
            os.environ.pop(var, None)


_sanitize_broken_proxy_env()


class Settings(BaseSettings):
    llm_provider: str = "apifreellm"

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    
    apifree_api_key: str = ""

    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 5

    vectorstore_path: str = "./data/vectorstore"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure vectorstore directory exists
Path(settings.vectorstore_path).mkdir(parents=True, exist_ok=True)
