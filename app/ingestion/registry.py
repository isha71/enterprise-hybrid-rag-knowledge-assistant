import json
from dataclasses import asdict
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.models.domain import Chunk


class ChunkRegistry:
    """Persistent chunk storage backed by a JSONL file.

    All indexed chunks are kept here so BM25 can rebuild its index on startup
    and so we can list documents without querying Qdrant.
    """

    def __init__(self, path: str | None = None):
        self.path = Path(path or settings.chunk_registry_path)
        self.chunks: list[Chunk] = []
        self._doc_map: dict[str, list[Chunk]] = {}

    def load(self) -> None:
        """Load chunks from the JSONL file if it exists."""
        self.chunks = []
        self._doc_map = {}

        if not self.path.exists():
            logger.info("No chunk registry found, starting fresh")
            return

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                chunk = Chunk(**data)
                self.chunks.append(chunk)
                self._doc_map.setdefault(chunk.document_id, []).append(chunk)

        logger.info(f"Loaded {len(self.chunks)} chunks for {len(self._doc_map)} documents")

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Append chunks to the registry and persist to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk)) + "\n")
                self.chunks.append(chunk)
                self._doc_map.setdefault(chunk.document_id, []).append(chunk)

    def get_documents(self) -> list[dict]:
        """Return a summary of all indexed documents."""
        docs = []
        for doc_id, doc_chunks in self._doc_map.items():
            first = doc_chunks[0]
            docs.append({
                "document_id": doc_id,
                "document_name": first.document_name,
                "file_type": first.file_type,
                "chunk_count": len(doc_chunks),
                "created_at": first.created_at,
            })
        return docs

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None


chunk_registry = ChunkRegistry()
