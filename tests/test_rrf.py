"""Tests for Reciprocal Rank Fusion."""

from app.models.domain import RetrievalResult
from app.retrieval.hybrid import rrf_fusion


def _make_result(chunk_id: str, rank: int, method: str, score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        score=score,
        rank=rank,
        retrieval_method=method,
        document_name="test.pdf",
        page_number=1,
    )


def test_rrf_basic_fusion():
    """Given known rankings from two sources, verify fused ordering."""
    dense = [
        _make_result("A", rank=1, method="dense"),
        _make_result("B", rank=2, method="dense"),
        _make_result("C", rank=3, method="dense"),
    ]
    bm25 = [
        _make_result("B", rank=1, method="bm25"),
        _make_result("C", rank=2, method="bm25"),
        _make_result("D", rank=3, method="bm25"),
    ]

    fused = rrf_fusion(dense, bm25, k=60)

    # B should be ranked first (appears in both lists at good ranks)
    assert fused[0].chunk_id == "B"
    # All unique chunks should appear
    fused_ids = [r.chunk_id for r in fused]
    assert set(fused_ids) == {"A", "B", "C", "D"}


def test_rrf_deduplication():
    """Verify that duplicate chunks are merged, not duplicated."""
    dense = [
        _make_result("X", rank=1, method="dense"),
        _make_result("Y", rank=2, method="dense"),
    ]
    bm25 = [
        _make_result("X", rank=1, method="bm25"),
        _make_result("Z", rank=2, method="bm25"),
    ]

    fused = rrf_fusion(dense, bm25, k=60)

    ids = [r.chunk_id for r in fused]
    assert len(ids) == len(set(ids)), "Duplicates found in fused results"


def test_rrf_scores_are_correct():
    """Verify RRF scores match the formula: Σ 1/(k + rank)."""
    k = 60

    dense = [_make_result("A", rank=1, method="dense")]
    bm25 = [_make_result("A", rank=2, method="bm25")]

    fused = rrf_fusion(dense, bm25, k=k)

    expected_score = 1 / (k + 1) + 1 / (k + 2)
    assert abs(fused[0].score - expected_score) < 1e-9


def test_rrf_single_list():
    """RRF with a single list should preserve ordering."""
    results = [
        _make_result("A", rank=1, method="dense"),
        _make_result("B", rank=2, method="dense"),
        _make_result("C", rank=3, method="dense"),
    ]

    fused = rrf_fusion(results, k=60)

    assert fused[0].chunk_id == "A"
    assert fused[1].chunk_id == "B"
    assert fused[2].chunk_id == "C"


def test_rrf_empty_lists():
    """RRF with empty lists should return empty results."""
    fused = rrf_fusion([], [], k=60)
    assert len(fused) == 0


def test_rrf_results_labeled_hybrid():
    """Verify fused results are labeled with 'hybrid' retrieval method."""
    dense = [_make_result("A", rank=1, method="dense")]
    bm25 = [_make_result("B", rank=1, method="bm25")]

    fused = rrf_fusion(dense, bm25, k=60)

    for result in fused:
        assert result.retrieval_method == "hybrid"
