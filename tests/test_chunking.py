"""Tests for chunking and metadata preservation."""

from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_text


def test_chunk_preserves_page_number():
    """Verify that page metadata survives splitting."""
    pages = [
        {"page_number": 1, "text": "This is content from page one. " * 50},
        {"page_number": 2, "text": "This is content from page two. " * 50},
    ]

    chunks = chunk_pages(
        pages=pages,
        document_id="test-doc-001",
        document_name="test.pdf",
        file_type=".pdf",
    )

    assert len(chunks) > 0

    # Every chunk should have a page number from the original pages
    for chunk in chunks:
        assert chunk.page_number in [1, 2]
        assert chunk.document_id == "test-doc-001"
        assert chunk.document_name == "test.pdf"


def test_chunk_preserves_document_metadata():
    """Verify that document-level metadata is on every chunk."""
    pages = [{"page_number": 5, "text": "Some document content. " * 30}]

    chunks = chunk_pages(
        pages=pages,
        document_id="doc-abc",
        document_name="policy.pdf",
        file_type=".pdf",
    )

    for chunk in chunks:
        assert chunk.document_id == "doc-abc"
        assert chunk.document_name == "policy.pdf"
        assert chunk.file_type == ".pdf"
        assert chunk.page_number == 5


def test_chunk_index_is_sequential():
    """Verify that chunk_index is sequential starting from 0."""
    pages = [
        {"page_number": 1, "text": "Content for page one. " * 50},
        {"page_number": 2, "text": "Content for page two. " * 50},
    ]

    chunks = chunk_pages(
        pages=pages,
        document_id="doc-seq",
        document_name="test.pdf",
        file_type=".pdf",
    )

    indices = [chunk.chunk_index for chunk in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_ids_are_unique():
    """Verify that every chunk has a unique ID."""
    pages = [
        {"page_number": 1, "text": "Some content. " * 50},
        {"page_number": 2, "text": "More content. " * 50},
    ]

    chunks = chunk_pages(
        pages=pages,
        document_id="doc-unique",
        document_name="test.pdf",
        file_type=".pdf",
    )

    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_clean_text_normalizes_whitespace():
    """Verify that cleaning normalizes whitespace without destroying content."""
    raw = "Hello    world.\n\n\n\nNext   paragraph."
    cleaned = clean_text(raw)

    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Hello" in cleaned
    assert "world" in cleaned
    assert "Next" in cleaned


def test_empty_pages_produce_no_chunks():
    """Verify that empty pages don't create chunks."""
    pages = [{"page_number": 1, "text": "   "}]

    chunks = chunk_pages(
        pages=pages,
        document_id="doc-empty",
        document_name="empty.pdf",
        file_type=".pdf",
    )

    assert len(chunks) == 0
