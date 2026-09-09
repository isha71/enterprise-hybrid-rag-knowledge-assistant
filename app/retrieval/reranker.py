from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import logger
from app.models.domain import RetrievalResult


class Reranker:
    """Cross-encoder reranker that rescores candidate chunks against the query."""

    def __init__(self):
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading reranker model: {settings.reranker_model}")
            self._model = CrossEncoder(settings.reranker_model)
            logger.info("Reranker model loaded")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates using cross-encoder scores.

        Scores each (query, chunk_text) pair and returns the top N by relevance.
        """
        top_n = top_n or settings.rerank_top_n

        if not candidates:
            return []

        # Build (query, text) pairs for the cross-encoder
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)

        # Attach rerank scores and sort
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        reranked = []
        for rank, (candidate, score) in enumerate(scored[:top_n], start=1):
            result = RetrievalResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=float(score),
                rank=rank,
                retrieval_method="hybrid_rerank",
                document_name=candidate.document_name,
                page_number=candidate.page_number,
                chunk_index=candidate.chunk_index,
                document_id=candidate.document_id,
            )
            reranked.append(result)

        logger.info(
            f"Reranked {len(candidates)} candidates → top {len(reranked)}"
        )
        return reranked


reranker = Reranker()
