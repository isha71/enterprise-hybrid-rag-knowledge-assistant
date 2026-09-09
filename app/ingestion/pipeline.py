import uuid
from pathlib import Path

from app.core.logging import logger
from app.ingestion.loaders import load_document
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import chunk_pages
from app.ingestion.registry import chunk_registry
from app.models.domain import Chunk


def ingest_document(file_path: Path, original_filename: str) -> tuple[str, list[Chunk]]:
    """Run the full ingestion pipeline for a single document.

    Returns (document_id, list of created chunks).

    Order: parse → clean → chunk → embed → Qdrant upsert → registry persist → BM25 rebuild.
    Registry is written only after Qdrant succeeds, so a Qdrant failure
    cannot leave a phantom entry in the lexical corpus.
    """
    document_id = str(uuid.uuid4())
    file_type = file_path.suffix.lower()

    logger.info(f"Ingesting document: {original_filename} (id={document_id})")

    # Step 1 — Parse text
    pages = load_document(file_path)
    if not pages:
        raise ValueError(f"No text could be extracted from {original_filename}")

    # Step 2 — Clean text
    for page in pages:
        page["text"] = clean_text(page["text"])

    pages = [p for p in pages if p["text"].strip()]
    if not pages:
        raise ValueError(f"Document {original_filename} contained no useful text after cleaning")

    # Step 3 — Chunk text
    chunks = chunk_pages(
        pages=pages,
        document_id=document_id,
        document_name=original_filename,
        file_type=file_type,
    )

    # Step 4 — Generate embeddings
    from app.retrieval.dense import embedding_model

    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.encode(texts)

    # Step 5 — Upsert to Qdrant (must succeed before persisting registry)
    from app.vectorstore.qdrant_store import qdrant_store

    qdrant_store.upsert_chunks(chunks, embeddings)

    # Step 6 — Persist to chunk registry (only after Qdrant succeeds)
    try:
        chunk_registry.add_chunks(chunks)
    except Exception as e:
        logger.error(
            f"Chunk registry write failed after Qdrant upsert for {original_filename}. "
            f"Qdrant has the vectors but the JSONL registry is inconsistent: {e}"
        )
        raise

    # Step 7 — Rebuild BM25 index
    from app.retrieval.bm25 import bm25_retriever

    bm25_retriever.rebuild(chunk_registry.chunks)

    logger.info(f"Ingestion complete: {original_filename} → {len(chunks)} chunks")
    return document_id, chunks
