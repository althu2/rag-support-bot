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
import requests
from typing import Any, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

logger = get_logger(__name__)

class ChatAPIFree(BaseChatModel):
    api_key: str
    model_name: str = "apifreellm"
    
    @property
    def _llm_type(self) -> str:
        return "apifree"
        
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        prompt = ""
        for m in messages:
            role = "User" if m.type == "human" else "Assistant" if m.type == "ai" else "System"
            prompt += f"{role}:\n{m.content}\n\n"
        
        prompt += "Assistant:\n"
        
        import time
        
        max_retries = 2
        for attempt in range(max_retries):
            response = requests.post(
                "https://apifreellm.com/api/v1/chat",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "message": prompt.strip(),
                    "model": self.model_name
                }
            )
            
            if response.status_code == 429:
                try:
                    data = response.json()
                    retry_after = data.get("retryAfter", 10)
                except:
                    retry_after = 10
                logger.warning(f"Rate limited (429). Sleeping for {retry_after}s before retry...")
                time.sleep(retry_after)
                continue
                
            if response.status_code >= 500:
                logger.warning(f"Server error {response.status_code}. Retrying...")
                time.sleep(2)
                continue
                
            response.raise_for_status()
            break
            
        data = response.json()
        
        if not data.get("success"):
            raise Exception(f"APIFreeLLM Error: {data.get('error', 'Unknown error')}")
            
        message = AIMessage(content=data.get("response", ""))
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful customer support assistant.
Answer the user's question using the context provided below.
If the answer is not in the context, say: "I'm sorry, I don't have information about that in the uploaded documents."
You may summarize the context if the user asks what the document is about or asks for a general summary.
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


def _ensure_apifree_key() -> None:
    key = (settings.apifree_api_key or "").strip()
    if not key:
        raise ValueError(
            "APIFREE_API_KEY is not configured. Set a real key in .env and restart the backend."
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

    if provider == "apifreellm":
        _ensure_apifree_key()
        # Fallback sequence of available free models under APIFreeLLM
        models = ["apifreellm", "llama-3", "mistral", "gemma"]
        main_llm = ChatAPIFree(api_key=settings.apifree_api_key, model_name=models[0])
        fallbacks = [ChatAPIFree(api_key=settings.apifree_api_key, model_name=m) for m in models[1:]]
        
        # Add One LLM API (llm7.io) as the ultimate fallback
        llm7_fallback = ChatOpenAI(
            model="gpt-4o-mini", # Defaults to a reliable model on their side
            api_key="sk-anonymous",
            base_url="https://api.llm7.io/v1",
            temperature=0.2,
            max_retries=2
        )
        fallbacks.append(llm7_fallback)
        
        return main_llm.with_fallbacks(fallbacks)

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini', 'openai', or 'apifreellm'."
    )


def _supports_system_role() -> bool:
    provider = (settings.llm_provider or "gemini").strip().lower()
    if provider == "apifreellm":
        return True
        
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
