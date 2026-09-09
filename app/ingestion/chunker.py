from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import logger
from app.models.domain import Chunk


def chunk_pages(
    pages: list[dict],
    document_id: str,
    document_name: str,
    file_type: str,
) -> list[Chunk]:
    """Split pages into overlapping chunks, preserving page metadata.

    Each page is chunked independently so page_number is always accurate.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    chunk_index = 0

    for page in pages:
        page_text = page["text"]
        page_number = page["page_number"]

        splits = splitter.split_text(page_text)

        for split_text in splits:
            if not split_text.strip():
                continue

            chunk = Chunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                chunk_index=chunk_index,
                text=split_text,
                file_type=file_type,
            )
            chunks.append(chunk)
            chunk_index += 1

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages ({document_name})")
    return chunks
