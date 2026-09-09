import time

from fastapi import APIRouter, HTTPException

from app.core.exceptions import VectorStoreError
from app.core.logging import logger
from app.ingestion.registry import chunk_registry
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrievedChunk,
    RetrieveResponse,
    TimingsMs,
)
from app.generation.rag_pipeline import query as rag_query, retrieve

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Ask a question against indexed documents using the specified retrieval strategy.

    Sync handler so CPU-heavy model inference runs in FastAPI's threadpool.
    """
    if not chunk_registry.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload documents first.",
        )

    try:
        response = rag_query(
            question=request.question,
            strategy=request.strategy,
            top_k=request.top_k,
        )
        return response
    except VectorStoreError as e:
        logger.error(f"Vector store error during query: {e}")
        raise HTTPException(status_code=503, detail="Vector store is unavailable")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(request: RetrieveRequest):
    """Retrieve chunks without calling the LLM. Useful for debugging and evaluation.

    Sync handler so CPU-heavy model inference runs in FastAPI's threadpool.
    """
    if not chunk_registry.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload documents first.",
        )

    try:
        t0 = time.time()
        results, candidate_count, retrieval_ms, reranking_ms = retrieve(
            question=request.question,
            strategy=request.strategy,
            top_k=request.top_k,
        )
        total_ms = (time.time() - t0) * 1000

        chunks = [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                document_name=r.document_name,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                text=r.text,
                score=round(r.score, 6),
                rank=r.rank,
                retrieval_method=r.retrieval_method,
            )
            for r in results
        ]

        return RetrieveResponse(
            question=request.question,
            strategy=request.strategy.value,
            chunks=chunks,
            timings_ms=TimingsMs(
                retrieval=round(retrieval_ms, 1),
                reranking=round(reranking_ms, 1),
                generation=0,
                total=round(total_ms, 1),
            ),
        )
    except VectorStoreError as e:
        logger.error(f"Vector store error during retrieve: {e}")
        raise HTTPException(status_code=503, detail="Vector store is unavailable")
    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
