"""Tests for the targeted fix pass: error handling, filenames, diagnostics, evaluation precision."""

import uuid
import numpy as np
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import VectorStoreError
from app.models.domain import RetrievalResult
from app.evaluation.metrics import hit_at_k, reciprocal_rank


# ── Evaluation relevance: page-level and evidence-level matching ──────

def test_evaluation_rejects_wrong_page_from_correct_document():
    """A chunk from page 3 must NOT count as a hit when relevant_source is page 7."""
    results = [
        RetrievalResult(
            chunk_id="c1", text="irrelevant chunk", score=0.9, rank=1,
            retrieval_method="dense", document_name="employee_handbook.pdf",
            page_number=3,
        ),
    ]
    relevant = [{"document_name": "employee_handbook.pdf", "page_number": 7}]

    assert hit_at_k(results, relevant, k=5) == 0.0
    assert reciprocal_rank(results, relevant) == 0.0


def test_evaluation_accepts_correct_page():
    """A chunk from the correct page (no text_contains) must count as a hit."""
    results = [
        RetrievalResult(
            chunk_id="c1", text="relevant chunk", score=0.9, rank=1,
            retrieval_method="dense", document_name="employee_handbook.pdf",
            page_number=7,
        ),
    ]
    relevant = [{"document_name": "employee_handbook.pdf", "page_number": 7}]

    assert hit_at_k(results, relevant, k=5) == 1.0
    assert reciprocal_rank(results, relevant) == 1.0


def test_evidence_rejects_correct_page_without_evidence():
    """Correct document + correct page but missing evidence text → NOT relevant."""
    results = [
        RetrievalResult(
            chunk_id="c1",
            text="Performance reviews are conducted annually during March.",
            score=0.9, rank=1,
            retrieval_method="dense", document_name="employee_handbook.pdf",
            page_number=8,
        ),
    ]
    relevant = [{
        "document_name": "employee_handbook.pdf",
        "page_number": 8,
        "text_contains": "Financial records: retained for 10 years",
    }]

    assert hit_at_k(results, relevant, k=5) == 0.0
    assert reciprocal_rank(results, relevant) == 0.0


def test_evidence_accepts_correct_page_with_evidence():
    """Correct document + correct page + evidence present → RELEVANT."""
    results = [
        RetrievalResult(
            chunk_id="c1",
            text="The company retains data. Financial records: retained for 10 years.",
            score=0.9, rank=1,
            retrieval_method="dense", document_name="employee_handbook.pdf",
            page_number=8,
        ),
    ]
    relevant = [{
        "document_name": "employee_handbook.pdf",
        "page_number": 8,
        "text_contains": "Financial records: retained for 10 years",
    }]

    assert hit_at_k(results, relevant, k=5) == 1.0
    assert reciprocal_rank(results, relevant) == 1.0


def test_evidence_matching_is_case_insensitive():
    """Evidence matching should tolerate case differences."""
    results = [
        RetrievalResult(
            chunk_id="c1",
            text="the it help desk is available monday to friday.",
            score=0.9, rank=1,
            retrieval_method="dense", document_name="employee_handbook.pdf",
            page_number=7,
        ),
    ]
    relevant = [{
        "document_name": "employee_handbook.pdf",
        "page_number": 7,
        "text_contains": "IT Help Desk",
    }]

    assert hit_at_k(results, relevant, k=5) == 1.0


# ── Fix #5: Qdrant search exception surfaces as VectorStoreError ─────

def test_qdrant_search_failure_raises_vectorstore_error():
    """Qdrant failure must raise VectorStoreError, not return empty list."""
    from app.vectorstore.qdrant_store import QdrantStore

    store = QdrantStore()
    # Simulate a broken client
    mock_client = MagicMock()
    mock_client.query_points.side_effect = ConnectionError("Qdrant unavailable")
    store._client = mock_client
    store._collection_name = "test"

    with pytest.raises(VectorStoreError, match="Qdrant search failed"):
        store.search([0.1, 0.2, 0.3], top_k=5)


# ── Fix #7: candidate_count reflects pre-reranking candidates ────────

def test_candidate_count_before_reranking():
    """retrieve() must return candidate_count BEFORE top-k / reranking."""
    from app.models.schemas import RetrievalStrategy

    # Create mock results simulating 15 hybrid candidates reranked to 5
    mock_hybrid_results = [
        RetrievalResult(
            chunk_id=f"c{i}", text=f"chunk {i}", score=0.5, rank=i,
            retrieval_method="hybrid", document_name="doc.pdf", page_number=1,
        )
        for i in range(1, 16)
    ]

    mock_reranked = mock_hybrid_results[:5]
    for i, r in enumerate(mock_reranked):
        r.rank = i + 1
        r.retrieval_method = "hybrid_rerank"

    with patch("app.generation.rag_pipeline.hybrid_retriever") as mock_hr, \
         patch("app.generation.rag_pipeline.reranker") as mock_rr:
        mock_hr.search.return_value = mock_hybrid_results
        mock_rr.rerank.return_value = mock_reranked

        from app.generation.rag_pipeline import retrieve

        results, candidate_count, _, _ = retrieve(
            "test question", RetrievalStrategy.HYBRID_RERANK, top_k=5
        )

        assert candidate_count == 15
        assert len(results) == 5


# ── Fix #11: Filename sanitization ───────────────────────────────────

def test_sanitize_strips_path_traversal():
    """Path traversal in filenames must be neutralized."""
    from app.api.routes.documents import _sanitize_filename

    storage, original = _sanitize_filename("../../etc/passwd.pdf")
    assert "/" not in storage
    assert ".." not in storage
    assert original == "passwd.pdf"


def test_sanitize_rejects_empty_filename():
    """Empty or dot-only filenames must be rejected."""
    from app.api.routes.documents import _sanitize_filename

    with pytest.raises(ValueError):
        _sanitize_filename("")

    with pytest.raises(ValueError):
        _sanitize_filename(".hidden")


def test_sanitize_unique_storage_names():
    """Two calls with the same filename must produce different storage names."""
    from app.api.routes.documents import _sanitize_filename

    s1, _ = _sanitize_filename("report.pdf")
    s2, _ = _sanitize_filename("report.pdf")
    assert s1 != s2


# ── Fix #12: Qdrant point IDs are deterministic UUIDs ────────────────

def test_qdrant_point_id_is_deterministic():
    """Same chunk_id must always produce the same Qdrant point ID."""
    from app.vectorstore.qdrant_store import _chunk_point_id

    id1 = _chunk_point_id("doc1_chunk_0")
    id2 = _chunk_point_id("doc1_chunk_0")
    id3 = _chunk_point_id("doc1_chunk_1")

    assert id1 == id2
    assert id1 != id3
    # Must be a valid UUID string
    uuid.UUID(id1)


# ── Smoke test: FastAPI app starts and serves /health ─────────────────

def test_fastapi_health_endpoint():
    """Smoke test that the app starts and /health returns 200."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "documents_indexed" in data
    assert "chunks_indexed" in data


def test_fastapi_query_requires_documents():
    """Query endpoint should return 400 when no documents are indexed."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/retrieve",
        json={"question": "test", "strategy": "vector"},
    )
    # Should get 400 because no documents are indexed in test env
    assert response.status_code == 400
    assert "indexed" in response.json()["detail"].lower()


# ── LLM provider selection ───────────────────────────────────────────

def test_llm_provider_ollama_dispatches_correctly():
    """When provider is 'ollama', generate() calls the Ollama path."""
    from app.generation.llm import LLMClient

    client = LLMClient()

    with patch.object(client, "_generate_ollama", return_value="ollama answer") as mock_ol, \
         patch("app.generation.llm.settings") as mock_settings:
        mock_settings.llm_provider = "ollama"
        mock_settings.llm_model = "qwen3:8b"
        result = client.generate("system", "user")

    mock_ol.assert_called_once_with("system", "user")
    assert result == "ollama answer"


def test_llm_provider_openai_dispatches_correctly():
    """When provider is 'openai', generate() calls the OpenAI path."""
    from app.generation.llm import LLMClient

    client = LLMClient()

    with patch.object(client, "_generate_openai", return_value="openai answer") as mock_oa, \
         patch("app.generation.llm.settings") as mock_settings:
        mock_settings.llm_provider = "openai"
        mock_settings.llm_model = "gpt-4o-mini"
        result = client.generate("system", "user")

    mock_oa.assert_called_once_with("system", "user")
    assert result == "openai answer"


def test_llm_provider_invalid_raises():
    """An unsupported provider must raise ValueError."""
    from app.generation.llm import LLMClient

    client = LLMClient()

    with patch("app.generation.llm.settings") as mock_settings:
        mock_settings.llm_provider = "anthropic"
        mock_settings.llm_model = "claude"

        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            client.generate("system", "user")


def test_ollama_no_openai_key_required():
    """Ollama provider must not require OPENAI_API_KEY."""
    from app.generation.llm import LLMClient

    client = LLMClient()

    with patch.object(client, "_generate_ollama", return_value="ok") as mock_ol, \
         patch("app.generation.llm.settings") as mock_settings:
        mock_settings.llm_provider = "ollama"
        mock_settings.llm_model = "qwen3:8b"
        mock_settings.openai_api_key = ""
        result = client.generate("s", "u")

    assert result == "ok"


# ── Citation verification ─────────────────────────────────────────────

def _make_result(chunk_id, text, rank):
    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=0.5, rank=rank,
        retrieval_method="hybrid_rerank", document_name="doc.pdf", page_number=1,
    )


def test_citation_verification_corrects_wrong_label():
    """LLM cited [S2] but S1 has the best cross-encoder score → corrected to [S1]."""
    from app.generation.citations import verify_citations

    results = [
        _make_result("c1", "E104 - License Expired: The platform license has expired. Read-only for 7 days.", 1),
        _make_result("c2", "E100 - Connection Timeout. E101 - Authentication Failed. E102 - Schema Mismatch.", 2),
    ]

    answer = "Error code E104 means the platform license has expired and the system remains read-only for 7 days [S2]."

    with patch("app.generation.citations.reranker") as mock_rr:
        mock_rr.model.predict.return_value = np.array([0.95, 0.30])
        corrected = verify_citations(answer, results)

    assert corrected.endswith("[S1].")
    assert "[S2]" not in corrected


def test_citation_verification_preserves_correct_labels():
    """Correctly cited bullets should keep their original labels with punctuation."""
    from app.generation.citations import verify_citations

    results = [
        _make_result("c1", "Employees may work remotely up to three days per week with manager approval.", 1),
        _make_result("c2", "Maintain regular working hours 9:00 AM to 5:30 PM. Use the company VPN.", 2),
    ]

    answer = (
        "Work from home guidelines:\n"
        "- Employees may work remotely up to three days per week [S1].\n"
        "- Maintain regular working hours and use the company VPN [S2]."
    )

    def mock_predict(pairs):
        claim = pairs[0][0]
        if "remotely" in claim:
            return np.array([0.9, 0.2])
        return np.array([0.1, 0.9])

    with patch("app.generation.citations.reranker") as mock_rr:
        mock_rr.model.predict.side_effect = mock_predict
        corrected = verify_citations(answer, results)

    lines = corrected.strip().split("\n")
    assert lines[1].endswith("[S1].")
    assert lines[2].endswith("[S2].")


def test_citation_verification_removes_invalid_labels():
    """Labels outside the source set (e.g. [S9] when only S1-S5 exist) must be removed."""
    from app.generation.citations import verify_citations

    results = [
        _make_result(f"c{i}", f"chunk {i}", i)
        for i in range(1, 6)
    ]

    answer = "Some claim with an invented citation [S9]."

    with patch("app.generation.citations.reranker") as mock_rr:
        mock_rr.model.predict.return_value = np.array([0.8, 0.1, 0.05, 0.02, 0.01])
        corrected = verify_citations(answer, results)

    assert "[S9]" not in corrected
    assert corrected.endswith("[S1].")


def test_citation_verification_no_labels_passthrough():
    """Lines without citation labels should pass through unchanged."""
    from app.generation.citations import verify_citations

    results = [_make_result("c1", "chunk 1", 1)]
    answer = "This answer has no citations at all."

    corrected = verify_citations(answer, results)
    assert corrected == answer


def test_citation_verification_without_terminal_punctuation():
    """Citation at end of line with no trailing punctuation."""
    from app.generation.citations import verify_citations

    results = [
        _make_result("c1", "The warranty covers manufacturing defects for 3 years", 1),
        _make_result("c2", "Extended warranty plans are available", 2),
    ]

    answer = "The warranty covers manufacturing defects [S2]"

    with patch("app.generation.citations.reranker") as mock_rr:
        mock_rr.model.predict.return_value = np.array([0.9, 0.2])
        corrected = verify_citations(answer, results)

    assert corrected == "The warranty covers manufacturing defects [S1]"
