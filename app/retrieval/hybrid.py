from app.core.config import settings
from app.core.logging import logger
from app.models.domain import RetrievalResult
from app.retrieval.dense import dense_retriever
from app.retrieval.bm25 import bm25_retriever


def rrf_fusion(
    *ranked_lists: list[RetrievalResult],
    k: int | None = None,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    RRF_score(d) = Σ 1 / (k + rank(d))

    This merges results from different retrieval methods onto a common scale,
    avoiding the problem of incomparable raw scores between dense and BM25.
    """
    k = k or settings.rrf_k
    scores: dict[str, float] = {}
    result_map: dict[str, RetrievalResult] = {}

    for ranked_list in ranked_lists:
        for result in ranked_list:
            chunk_id = result.chunk_id
            rrf_score = 1.0 / (k + result.rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

            # Keep the result object (prefer first occurrence for metadata)
            if chunk_id not in result_map:
                result_map[chunk_id] = result

    # Sort by fused RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused_results = []
    for rank, chunk_id in enumerate(sorted_ids, start=1):
        original = result_map[chunk_id]
        fused = RetrievalResult(
            chunk_id=chunk_id,
            text=original.text,
            score=scores[chunk_id],
            rank=rank,
            retrieval_method="hybrid",
            document_name=original.document_name,
            page_number=original.page_number,
            chunk_index=original.chunk_index,
            document_id=original.document_id,
        )
        fused_results.append(fused)

    return fused_results


class HybridRetriever:
    """Runs dense + BM25 retrieval and fuses results with RRF."""

    def search(
        self,
        query: str,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
    ) -> list[RetrievalResult]:
        dense_results = dense_retriever.search(query, top_k=dense_top_k)
        bm25_results = bm25_retriever.search(query, top_k=bm25_top_k)

        logger.info(
            f"Hybrid retrieval: {len(dense_results)} dense + {len(bm25_results)} BM25 results"
        )

        fused = rrf_fusion(dense_results, bm25_results)
        logger.info(f"RRF fusion produced {len(fused)} unique candidates")

        return fused


hybrid_retriever = HybridRetriever()
