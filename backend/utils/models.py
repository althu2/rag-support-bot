from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "human" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Each tuple is (human_message, ai_message)
    history: List[Tuple[str, str]] = Field(default_factory=list)


class SourceReference(BaseModel):
    source: str
    page: int | str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceReference]


class UploadResponse(BaseModel):
    message: str
    filename: str
    pages_processed: int
    chunks_created: int


class StatusResponse(BaseModel):
    vectorstore_loaded: bool
    document_count: int
