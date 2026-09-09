"""Application-specific exceptions."""


class VectorStoreError(Exception):
    """Raised when Qdrant or vector store operations fail unexpectedly.

    Distinguishes infrastructure failures from legitimate empty search results.
    """
