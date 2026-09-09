from pathlib import Path

from pypdf import PdfReader

from app.core.logging import logger


def load_pdf(file_path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns list of {page_number, text}."""
    reader = PdfReader(str(file_path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_number": i + 1, "text": text})

    logger.info(f"Extracted {len(pages)} pages from PDF: {file_path.name}")
    return pages


def load_text(file_path: Path) -> list[dict]:
    """Read a plain text file. Returns list with single entry (no page concept)."""
    text = file_path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        logger.warning(f"Empty text file: {file_path.name}")
        return []

    logger.info(f"Loaded text file: {file_path.name}")
    return [{"page_number": None, "text": text}]


def load_markdown(file_path: Path) -> list[dict]:
    """Read a markdown file. Returns list with single entry."""
    text = file_path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        logger.warning(f"Empty markdown file: {file_path.name}")
        return []

    logger.info(f"Loaded markdown file: {file_path.name}")
    return [{"page_number": None, "text": text}]


SUPPORTED_LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_markdown,
}


def load_document(file_path: Path) -> list[dict]:
    """Load a document based on file extension. Returns list of {page_number, text}."""
    ext = file_path.suffix.lower()
    loader = SUPPORTED_LOADERS.get(ext)

    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(SUPPORTED_LOADERS.keys())}")

    return loader(file_path)
