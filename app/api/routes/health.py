from fastapi import APIRouter

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Return service status. Lightweight, no blocking work."""
    from app.ingestion.registry import chunk_registry

    documents = set()
    for chunk in chunk_registry.chunks:
        documents.add(chunk.document_id)

    return HealthResponse(
        status="ok",
        documents_indexed=len(documents),
        chunks_indexed=len(chunk_registry.chunks),
    )
