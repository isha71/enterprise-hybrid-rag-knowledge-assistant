import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.loaders import SUPPORTED_LOADERS
from app.ingestion.pipeline import ingest_document
from app.ingestion.registry import chunk_registry
from app.models.schemas import (
    DocumentInfo,
    DocumentListItem,
    DocumentListResponse,
    UploadResponse,
)

router = APIRouter()


def _sanitize_filename(raw: str) -> tuple[str, str]:
    """Sanitize an uploaded filename.

    Returns (safe_storage_name, original_document_name).
    Strips path components to prevent traversal, and prefixes with a UUID
    so two uploads with the same name don't overwrite each other.
    """
    # Strip any path components (prevents ../../ traversal)
    clean_name = Path(raw).name

    if not clean_name or clean_name.startswith("."):
        raise ValueError(f"Invalid filename: {raw}")

    storage_name = f"{uuid.uuid4().hex[:12]}_{clean_name}"
    return storage_name, clean_name


@router.post("/documents/upload", response_model=UploadResponse)
def upload_documents(files: list[UploadFile] = File(...)):
    """Upload one or more documents for ingestion.

    Sync handler so CPU-heavy ingestion (PDF parsing, embedding, Qdrant upsert)
    runs in FastAPI's threadpool instead of blocking the async event loop.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File has no filename")

        # Sanitize filename
        try:
            storage_name, original_name = _sanitize_filename(file.filename)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filename: {file.filename}",
            )

        ext = Path(original_name).suffix.lower()
        if ext not in SUPPORTED_LOADERS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: {list(SUPPORTED_LOADERS.keys())}",
            )

        # Save uploaded file with safe storage name
        file_path = upload_dir / storage_name
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            logger.error(f"Failed to save file {original_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {original_name}")

        # Run ingestion pipeline (uses original_name for metadata/citations)
        try:
            document_id, chunks = ingest_document(file_path, original_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Ingestion failed for {original_name}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Ingestion failed for {original_name}: {str(e)}"
            )

        results.append(
            DocumentInfo(
                document_id=document_id,
                document_name=original_name,
                chunks_created=len(chunks),
            )
        )

    return UploadResponse(documents=results)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """List all indexed documents."""
    docs = chunk_registry.get_documents()
    items = [
        DocumentListItem(
            document_id=d["document_id"],
            document_name=d["document_name"],
            file_type=d["file_type"],
            chunk_count=d["chunk_count"],
            created_at=d["created_at"],
        )
        for d in docs
    ]
    return DocumentListResponse(documents=items)
