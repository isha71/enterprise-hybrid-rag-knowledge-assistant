from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise Hybrid RAG Knowledge Assistant")
    logger.info(f"Environment: {settings.app_env}")

    # Load chunk registry and rebuild BM25 on startup
    from app.ingestion.registry import chunk_registry

    chunk_registry.load()
    logger.info(f"Loaded {len(chunk_registry.chunks)} chunks from registry")

    from app.retrieval.bm25 import bm25_retriever

    bm25_retriever.rebuild(chunk_registry.chunks)
    logger.info("BM25 index rebuilt")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Enterprise Hybrid RAG Knowledge Assistant",
    description="A hybrid retrieval-augmented generation system with dense, BM25, and reranked retrieval strategies.",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routes
from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router
from app.api.routes.evaluation import router as evaluation_router

app.include_router(health_router)
app.include_router(documents_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")
