from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    chunk_index: int
    text: str
    file_type: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    rank: int
    retrieval_method: str
    document_name: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    document_id: str = ""
