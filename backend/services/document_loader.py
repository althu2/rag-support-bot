import fitz  # PyMuPDF
from pathlib import Path
from typing import List
from dataclasses import dataclass
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    page_number: int
    text: str
    source: str


def load_pdf(file_path: str) -> List[ParsedPage]:
    """
    Load a PDF and return a list of ParsedPage objects, one per page.
    Raises ValueError for empty or unreadable PDFs.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages: List[ParsedPage] = []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {e}")

    if len(doc) == 0:
        raise ValueError("PDF has no pages.")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if text:  # skip blank pages silently
            pages.append(
                ParsedPage(
                    page_number=page_num + 1,
                    text=text,
                    source=path.name,
                )
            )

    doc.close()

    if not pages:
        raise ValueError("PDF contains no extractable text (possibly scanned/image-based).")

    logger.info(f"Loaded '{path.name}' — {len(pages)} pages with text extracted.")
    return pages
