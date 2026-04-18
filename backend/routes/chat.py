from fastapi import APIRouter, HTTPException, Request
from backend.config import settings
from backend.services.rag_chain import rag_answer
from backend.utils.models import ChatRequest, ChatResponse, SourceReference
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


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


@router.post("/", response_model=ChatResponse)
async def chat(http_request: Request, request: ChatRequest):
    """
    Answer a question using RAG over indexed documents.
    Accepts conversation history for multi-turn support.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    _apply_runtime_settings(http_request)

    try:
        result = await rag_answer(
            question=request.question,
            chat_history=request.history,
        )

        sources = [
            SourceReference(source=s["source"], page=s["page"])
            for s in result["sources"]
        ]

        return ChatResponse(answer=result["answer"], sources=sources)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
