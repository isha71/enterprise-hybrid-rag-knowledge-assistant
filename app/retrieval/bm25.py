import re

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import logger
from app.models.domain import Chunk, RetrievalResult


def tokenize(text: str) -> list[str]:
    """Simple reproducible tokenizer: lowercase, split on whitespace, strip punctuation."""
    text = text.lower()
    # Remove punctuation except hyphens within words
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = text.split()
    # Strip leading/trailing hyphens from each token
    tokens = [t.strip("-") for t in tokens if t.strip("-")]
    return tokens


class BM25Retriever:
    """Lexical retriever using BM25Okapi over the chunk corpus."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Rebuild the BM25 index from the full chunk corpus."""
        self._chunks = list(chunks)

        if not self._chunks:
            self._bm25 = None
            self._tokenized_corpus = []
            logger.info("BM25 index cleared (no chunks)")
            return

        self._tokenized_corpus = [tokenize(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index rebuilt with {len(self._chunks)} chunks")

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or settings.bm25_top_k

        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 index is empty, returning no results")
            return []

        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top K indices sorted by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self._chunks[idx]
            score = float(scores[idx])

            # Skip zero-score results
            if score <= 0:
                continue

            result = RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
                rank=rank,
                retrieval_method="bm25",
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                document_id=chunk.document_id,
            )
            results.append(result)

        logger.info(f"BM25 retrieval returned {len(results)} results for query")
        return results


bm25_retriever = BM25Retriever()
