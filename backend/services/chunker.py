from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.services.document_loader import ParsedPage
from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_pages(pages: List[ParsedPage]) -> List[Document]:
    """
    Split parsed pages into overlapping chunks.
    Each chunk carries metadata: source filename and page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents: List[Document] = []

    for page in pages:
        chunks = splitter.split_text(page.text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": page.source,
                    "page": page.page_number,
                    "chunk_index": i,
                },
            )
            documents.append(doc)

    logger.info(f"Created {len(documents)} chunks from {len(pages)} pages.")
    return documents
