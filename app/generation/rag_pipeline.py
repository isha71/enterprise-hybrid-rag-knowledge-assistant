import time

from app.core.logging import logger
from app.models.domain import RetrievalResult
from app.models.schemas import (
    QueryResponse,
    RetrievalInfo,
    RetrievalStrategy,
    SourceInfo,
    TimingsMs,
)
from app.retrieval.dense import dense_retriever
from app.retrieval.hybrid import hybrid_retriever
from app.retrieval.reranker import reranker
from app.generation.prompts import SYSTEM_PROMPT, build_context, build_user_prompt
from app.generation.llm import llm_client
from app.generation.citations import verify_citations


def retrieve(
    question: str,
    strategy: RetrievalStrategy,
    top_k: int = 5,
) -> tuple[list[RetrievalResult], int, float, float]:
    """Run retrieval with the specified strategy.

    Returns (final_results, candidate_count, retrieval_time_ms, reranking_time_ms).
    candidate_count is the number of chunks BEFORE final top-k / reranking selection.
    """
    reranking_ms = 0.0

    t0 = time.time()

    if strategy == RetrievalStrategy.VECTOR:
        candidates = dense_retriever.search(question, top_k=top_k)
        retrieval_ms = (time.time() - t0) * 1000
        candidate_count = len(candidates)
        final_results = candidates[:top_k]

    elif strategy == RetrievalStrategy.HYBRID:
        candidates = hybrid_retriever.search(question)
        retrieval_ms = (time.time() - t0) * 1000
        candidate_count = len(candidates)
        final_results = candidates[:top_k]

    elif strategy == RetrievalStrategy.HYBRID_RERANK:
        candidates = hybrid_retriever.search(question)
        retrieval_ms = (time.time() - t0) * 1000
        candidate_count = len(candidates)

        t1 = time.time()
        final_results = reranker.rerank(question, candidates, top_n=top_k)
        reranking_ms = (time.time() - t1) * 1000

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return final_results, candidate_count, retrieval_ms, reranking_ms


def query(
    question: str,
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
    top_k: int = 5,
) -> QueryResponse:
    """Full RAG pipeline: retrieve → build context → generate answer → return structured response."""
    total_start = time.time()

    # Retrieval
    results, candidate_count, retrieval_ms, reranking_ms = retrieve(
        question, strategy, top_k
    )

    # Build context from top chunks
    context = build_context(results)

    # Generate answer
    t_gen = time.time()
    user_prompt = build_user_prompt(question, context)
    raw_answer = llm_client.generate(SYSTEM_PROMPT, user_prompt)
    answer = verify_citations(raw_answer, results)
    generation_ms = (time.time() - t_gen) * 1000

    total_ms = (time.time() - total_start) * 1000

    # Build source citations from retrieval metadata
    sources = [
        SourceInfo(
            source_id=f"S{i}",
            document_name=r.document_name,
            page_number=r.page_number,
            chunk_id=r.chunk_id,
            snippet=r.text[:200],
        )
        for i, r in enumerate(results, start=1)
    ]

    return QueryResponse(
        question=question,
        answer=answer,
        strategy=strategy.value,
        sources=sources,
        retrieval=RetrievalInfo(
            candidate_count=candidate_count,
            context_count=len(results),
        ),
        timings_ms=TimingsMs(
            retrieval=round(retrieval_ms, 1),
            reranking=round(reranking_ms, 1),
            generation=round(generation_ms, 1),
            total=round(total_ms, 1),
        ),
    )
