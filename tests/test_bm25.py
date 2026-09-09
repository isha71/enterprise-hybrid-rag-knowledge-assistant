"""Tests for BM25 retrieval."""

from app.models.domain import Chunk
from app.retrieval.bm25 import BM25Retriever, tokenize


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-test",
        document_name="test.pdf",
        page_number=1,
        chunk_index=0,
        text=text,
    )


def test_tokenize_lowercases():
    tokens = tokenize("Hello World")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_removes_punctuation():
    tokens = tokenize("Hello, world! How are you?")
    assert "hello" in tokens
    assert "world" in tokens
    # Commas and exclamation marks should not appear
    assert all("," not in t and "!" not in t and "?" not in t for t in tokens)


def test_bm25_exact_term_retrieval():
    """Verify that BM25 retrieves chunks containing exact query terms."""
    retriever = BM25Retriever()

    chunks = [
        _make_chunk("c1", "The employee handbook covers maternity leave policies in detail."),
        _make_chunk("c2", "Product XR-7000 specifications include 16GB RAM and SSD storage."),
        _make_chunk("c3", "Annual performance reviews are conducted every December."),
    ]

    retriever.rebuild(chunks)

    # Searching for a specific product code should rank c2 first
    results = retriever.search("XR-7000", top_k=3)
    assert len(results) > 0
    assert results[0].chunk_id == "c2"


def test_bm25_results_labeled():
    """Verify BM25 results are labeled with correct retrieval method."""
    retriever = BM25Retriever()
    chunks = [
        _make_chunk("c1", "Some text about policies and procedures for employees."),
        _make_chunk("c2", "Unrelated content about weather and climate patterns."),
        _make_chunk("c3", "Information about cooking recipes and kitchen tools."),
        _make_chunk("c4", "Discussion of sports events and fitness training."),
    ]
    retriever.rebuild(chunks)

    results = retriever.search("policies", top_k=1)
    assert len(results) > 0
    assert results[0].retrieval_method == "bm25"


def test_bm25_empty_corpus():
    """BM25 with no chunks should return empty results."""
    retriever = BM25Retriever()
    retriever.rebuild([])

    results = retriever.search("anything", top_k=5)
    assert len(results) == 0


def test_bm25_ranking_order():
    """Verify that more relevant documents rank higher."""
    retriever = BM25Retriever()

    chunks = [
        _make_chunk("c1", "cats dogs birds fish animals"),
        _make_chunk("c2", "machine learning artificial intelligence deep learning neural networks machine learning"),
        _make_chunk("c3", "cooking recipes pasta bread baking"),
    ]

    retriever.rebuild(chunks)

    results = retriever.search("machine learning", top_k=3)
    assert results[0].chunk_id == "c2"
