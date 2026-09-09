from enum import Enum

from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


# --- Document Schemas ---


class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    chunks_created: int


class UploadResponse(BaseModel):
    documents: list[DocumentInfo]


class DocumentListItem(BaseModel):
    document_id: str
    document_name: str
    file_type: str
    chunk_count: int
    created_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]


# --- Query Schemas ---


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID_RERANK,
        description="Retrieval strategy to use",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks")


class SourceInfo(BaseModel):
    source_id: str = Field(description="Stable label used in LLM context, e.g. S1")
    document_name: str
    page_number: int | None
    chunk_id: str
    snippet: str


class TimingsMs(BaseModel):
    retrieval: float = 0
    reranking: float = 0
    generation: float = 0
    total: float = 0


class RetrievalInfo(BaseModel):
    candidate_count: int = 0
    context_count: int = 0


class QueryResponse(BaseModel):
    question: str
    answer: str
    strategy: str
    sources: list[SourceInfo]
    retrieval: RetrievalInfo
    timings_ms: TimingsMs


# --- Retrieve Debug Schemas ---


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID_RERANK)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int | None
    chunk_index: int
    text: str
    score: float
    rank: int
    retrieval_method: str


class RetrieveResponse(BaseModel):
    question: str
    strategy: str
    chunks: list[RetrievedChunk]
    timings_ms: TimingsMs


# --- Evaluation Schemas ---


class EvaluationRequest(BaseModel):
    dataset_path: str = Field(default="evaluation/dataset.json")


class StrategyResult(BaseModel):
    strategy: str
    hit_rate_at_5: float
    mrr: float
    details: list[dict] = []


class EvaluationResponse(BaseModel):
    results: list[StrategyResult]


# --- Health ---


class HealthResponse(BaseModel):
    status: str = "ok"
    documents_indexed: int = 0
    chunks_indexed: int = 0
