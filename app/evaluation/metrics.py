from app.models.domain import RetrievalResult


def _is_relevant(result: RetrievalResult, source: dict) -> bool:
    """Check whether a single retrieved result matches a single relevance label.

    Matching requires:
      1. document_name matches
      2. page_number matches (if specified in source)
      3. text_contains evidence is found in chunk text (if specified in source)

    When text_contains is supplied, a correct-document-and-page chunk that
    does NOT contain the evidence string is NOT counted as relevant.
    This prevents an unrelated section on the same page from being a false hit.
    """
    if result.document_name != source["document_name"]:
        return False

    if source.get("page_number") is not None:
        if result.page_number != source["page_number"]:
            return False

    evidence = source.get("text_contains")
    if evidence is not None:
        if evidence.lower() not in result.text.lower():
            return False

    return True


def hit_at_k(
    results: list[RetrievalResult],
    relevant_sources: list[dict],
    k: int = 5,
) -> float:
    """Check if any relevant source appears in the first K results.

    Returns 1.0 if a hit is found, 0.0 otherwise.
    """
    for result in results[:k]:
        for source in relevant_sources:
            if _is_relevant(result, source):
                return 1.0

    return 0.0


def reciprocal_rank(
    results: list[RetrievalResult],
    relevant_sources: list[dict],
) -> float:
    """Compute 1 / rank_of_first_relevant_result.

    Returns 0.0 if no relevant result is found.
    """
    for result in results:
        for source in relevant_sources:
            if _is_relevant(result, source):
                return 1.0 / result.rank

    return 0.0
