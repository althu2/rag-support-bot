from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from backend.services.vector_store import load_vectorstore
from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful customer support assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say: "I'm sorry, I don't have information about that in the uploaded documents."
Do NOT make up information. Be concise and accurate.

Context:
{context}"""

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
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


def _get_llm():
    provider = (settings.llm_provider or "gemini").strip().lower()

    if provider == "openai":
        _ensure_openai_key()
        return ChatOpenAI(
            model=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.2,
            streaming=False,
        )

    if provider == "gemini":
        _ensure_gemini_key()
        return ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini' or 'openai'."
    )


def _supports_system_role() -> bool:
    provider = (settings.llm_provider or "gemini").strip().lower()
    model_name = (
        settings.gemini_chat_model if provider == "gemini" else settings.openai_chat_model
    )
    return not (provider == "gemini" and "gemma" in model_name.lower())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: List[Tuple[str, str]]):
    """Convert list of (human, ai) tuples to LangChain message objects."""
    messages = []
    for human, ai in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages


def _extract_sources(docs: List[Document]) -> List[dict]:
    seen = set()
    sources = []
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return sources


# ---------------------------------------------------------------------------
# Main RAG function
# ---------------------------------------------------------------------------
async def rag_answer(
    question: str,
    chat_history: List[Tuple[str, str]] = None,
) -> dict:
    """
    Run RAG pipeline. Returns answer text + source citations.
    """
    if chat_history is None:
        chat_history = []

    # 1. Load vectorstore
    store = load_vectorstore()
    if store is None:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "sources": [],
        }

    # 2. Retrieve relevant chunks
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "sources": [],
        }

    # 3. Build prompt with history
    if _supports_system_role():
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
    else:
        # Some Gemini-hosted Gemma models reject system/developer role messages.
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", SYSTEM_PROMPT + "\n\nUser question:\n{question}"),
            ]
        )

    # 4. LCEL chain: prompt | llm | parser
    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()

    # 5. Invoke
    answer = chain.invoke(
        {
            "context": _format_docs(docs),
            "chat_history": _format_history(chat_history),
            "question": question,
        }
    )

    sources = _extract_sources(docs)
    logger.info(f"RAG answered question. Sources: {sources}")

    return {"answer": answer, "sources": sources}
