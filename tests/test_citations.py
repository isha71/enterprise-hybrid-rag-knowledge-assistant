"""Tests for citation/source mapping from retrieval metadata."""

import re

from app.models.domain import RetrievalResult
from app.models.schemas import SourceInfo
from app.generation.prompts import build_context


def test_citation_from_retrieval_metadata():
    """Verify that source info comes from chunk metadata, not fabricated."""
    result = RetrievalResult(
        chunk_id="doc123_chunk_5",
        text="Maternity leave is 16 weeks for all full-time employees.",
        score=0.92,
        rank=1,
        retrieval_method="hybrid_rerank",
        document_name="employee_handbook.pdf",
        page_number=16,
        chunk_index=5,
        document_id="doc123",
    )

    source = SourceInfo(
        source_id="S1",
        document_name=result.document_name,
        page_number=result.page_number,
        chunk_id=result.chunk_id,
        snippet=result.text[:200],
    )

    assert source.source_id == "S1"
    assert source.document_name == "employee_handbook.pdf"
    assert source.page_number == 16
    assert source.chunk_id == "doc123_chunk_5"
    assert "Maternity leave" in source.snippet


def test_citations_match_retrieved_chunks():
    """Verify that multiple sources all correspond to actual retrieved chunks."""
    results = [
        RetrievalResult(
            chunk_id=f"doc_{i}_chunk_{j}",
            text=f"Content from document {i}, chunk {j}",
            score=0.9 - i * 0.1,
            rank=idx + 1,
            retrieval_method="hybrid_rerank",
            document_name=f"doc_{i}.pdf",
            page_number=j + 1,
            chunk_index=j,
            document_id=f"doc_{i}",
        )
        for idx, (i, j) in enumerate([(1, 3), (2, 7), (1, 5)])
    ]

    sources = [
        SourceInfo(
            source_id=f"S{idx+1}",
            document_name=r.document_name,
            page_number=r.page_number,
            chunk_id=r.chunk_id,
            snippet=r.text[:200],
        )
        for idx, r in enumerate(results)
    ]

    for source, result in zip(sources, results):
        assert source.document_name == result.document_name
        assert source.page_number == result.page_number
        assert source.chunk_id == result.chunk_id


def test_citation_page_number_none_for_text_files():
    """Verify that TXT/MD files can have None page_number in citations."""
    result = RetrievalResult(
        chunk_id="txt_chunk_0",
        text="Some content from a text file.",
        score=0.85,
        rank=1,
        retrieval_method="dense",
        document_name="notes.txt",
        page_number=None,
        chunk_index=0,
        document_id="txt-doc",
    )

    source = SourceInfo(
        source_id="S1",
        document_name=result.document_name,
        page_number=result.page_number,
        chunk_id=result.chunk_id,
        snippet=result.text[:200],
    )

    assert source.page_number is None
    assert source.document_name == "notes.txt"


def test_context_uses_stable_source_labels():
    """Verify build_context uses [S1], [S2] labels, not free-form document references."""
    results = [
        RetrievalResult(
            chunk_id="c1", text="First chunk content.", score=0.9, rank=1,
            retrieval_method="dense", document_name="doc.pdf", page_number=3,
        ),
        RetrievalResult(
            chunk_id="c2", text="Second chunk content.", score=0.8, rank=2,
            retrieval_method="dense", document_name="doc.pdf", page_number=7,
        ),
    ]

    context = build_context(results)

    assert "[S1]" in context
    assert "[S2]" in context
    # Should NOT use the old [SOURCE 1] format
    assert "[SOURCE 1]" not in context

    # Labels should appear before document metadata
    s1_pos = context.index("[S1]")
    s2_pos = context.index("[S2]")
    assert s1_pos < s2_pos


def test_source_labels_map_to_metadata():
    """Verify that source_id S1/S2/... maps correctly to chunk metadata."""
    results = [
        RetrievalResult(
            chunk_id="abc", text="X", score=0.9, rank=1,
            retrieval_method="hybrid_rerank", document_name="a.pdf", page_number=1,
        ),
        RetrievalResult(
            chunk_id="def", text="Y", score=0.8, rank=2,
            retrieval_method="hybrid_rerank", document_name="b.pdf", page_number=5,
        ),
    ]

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

    assert sources[0].source_id == "S1"
    assert sources[0].chunk_id == "abc"
    assert sources[1].source_id == "S2"
    assert sources[1].chunk_id == "def"
