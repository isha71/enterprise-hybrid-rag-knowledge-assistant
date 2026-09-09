from fastapi import APIRouter, HTTPException

from app.core.exceptions import VectorStoreError
from app.core.logging import logger
from app.ingestion.registry import chunk_registry
from app.models.schemas import EvaluationRequest, EvaluationResponse
from app.evaluation.evaluator import run_evaluation

router = APIRouter()


@router.post("/evaluation/run", response_model=EvaluationResponse)
def run_eval(request: EvaluationRequest):
    """Run retrieval evaluation across all three strategies.

    Sync handler so CPU-heavy model inference runs in FastAPI's threadpool.
    """
    if not chunk_registry.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload documents first.",
        )

    try:
        response = run_evaluation(request.dataset_path)
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VectorStoreError as e:
        logger.error(f"Vector store error during evaluation: {e}")
        raise HTTPException(status_code=503, detail="Vector store is unavailable")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
