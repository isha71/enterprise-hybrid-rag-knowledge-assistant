import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import logger
from app.models.domain import RetrievalResult
from app.vectorstore.qdrant_store import qdrant_store


class EmbeddingModel:
    """Wrapper around sentence-transformers for lazy loading."""

    def __init__(self):
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)
            logger.info("Embedding model loaded")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]


embedding_model = EmbeddingModel()


class DenseRetriever:
    """Retrieves chunks using vector similarity search against Qdrant."""

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or settings.dense_top_k

        query_vector = embedding_model.encode_query(query)
        raw_results = qdrant_store.search(query_vector.tolist(), top_k=top_k)

        results = []
        for rank, item in enumerate(raw_results, start=1):
            payload = item["payload"]
            result = RetrievalResult(
                chunk_id=payload["chunk_id"],
                text=payload["text"],
                score=item["score"],
                rank=rank,
                retrieval_method="dense",
                document_name=payload.get("document_name", ""),
                page_number=payload.get("page_number"),
                chunk_index=payload.get("chunk_index", 0),
                document_id=payload.get("document_id", ""),
            )
            results.append(result)

        logger.info(f"Dense retrieval returned {len(results)} results for query")
        return results


dense_retriever = DenseRetriever()
