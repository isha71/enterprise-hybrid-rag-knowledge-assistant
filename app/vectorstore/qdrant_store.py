import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import logger
from app.models.domain import Chunk


def _chunk_point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk's Qdrant point, derived from chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantStore:
    """Manages the Qdrant vector database connection and operations."""

    def __init__(self):
        self._client: QdrantClient | None = None
        self._collection_name = settings.qdrant_collection

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if settings.qdrant_url:
                self._client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                )
                logger.info(f"Connected to remote Qdrant at {settings.qdrant_url}")
            else:
                self._client = QdrantClient(path=settings.qdrant_dir)
                logger.info(f"Using local Qdrant storage at {settings.qdrant_dir}")
        return self._client

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self._collection_name not in collections:
            self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self._collection_name} (dim={vector_size})")

    def upsert_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Insert chunk vectors and metadata into Qdrant."""
        if len(chunks) == 0:
            return

        vector_size = embeddings.shape[1]
        self.ensure_collection(vector_size)

        points = []
        for i, chunk in enumerate(chunks):
            point = PointStruct(
                id=_chunk_point_id(chunk.chunk_id),
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "file_type": chunk.file_type,
                },
            )
            points.append(point)

        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self._collection_name,
                points=batch,
            )

        logger.info(f"Upserted {len(points)} vectors into Qdrant")

    def search(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        """Search for similar vectors. Returns list of {payload, score}.

        Raises VectorStoreError on infrastructure failure instead of
        silently returning an empty list.
        """
        try:
            results = self.client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            return [
                {"payload": point.payload, "score": point.score}
                for point in results.points
            ]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            raise VectorStoreError(f"Qdrant search failed: {e}") from e


qdrant_store = QdrantStore()
