"""Post-generation citation verification using the existing cross-encoder."""

import re

from app.core.logging import logger
from app.models.domain import RetrievalResult
from app.retrieval.reranker import reranker

# Matches citation labels like [S1], [S2], [S12]
_CITE_RE = re.compile(r"\[S(\d+)\]")


def verify_citations(answer: str, results: list[RetrievalResult]) -> str:
    """Verify and correct inline citation labels in the LLM answer.

    For each line/bullet that contains one or more [Sn] labels:
      1. Strip the labels to get the bare claim text.
      2. Score the claim against every retrieved chunk using the existing
         cross-encoder (reranker.model).
      3. Replace the label with the source whose chunk scores highest.

    Labels referencing sources outside the available set are removed.
    """
    if not results:
        return answer

    num_sources = len(results)
    lines = answer.split("\n")
    verified_lines: list[str] = []

    for line in lines:
        labels_in_line = _CITE_RE.findall(line)
        if not labels_in_line:
            verified_lines.append(line)
            continue

        # Strip all citation markers to get the bare claim
        claim = _CITE_RE.sub("", line).strip()
        if not claim:
            verified_lines.append(line)
            continue

        # Score claim against every retrieved chunk
        pairs = [(claim, r.text) for r in results]
        scores = reranker.model.predict(pairs)

        best_idx = int(scores.argmax())
        best_label = f"[S{best_idx + 1}]"

        # Rebuild the line: strip old labels, place the verified one before
        # any terminal punctuation so "claim [S2]." becomes "claim [S1]."
        cleaned = _CITE_RE.sub("", line).rstrip()

        trailing_punct = ""
        if cleaned and cleaned[-1] in ".,;:?!":
            trailing_punct = cleaned[-1]
            cleaned = cleaned[:-1].rstrip()

        verified_lines.append(f"{cleaned} {best_label}{trailing_punct}")

    corrected = "\n".join(verified_lines)

    if corrected != answer:
        logger.info("Citation verification adjusted inline labels")

    return corrected
