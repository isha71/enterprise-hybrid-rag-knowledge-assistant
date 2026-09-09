from app.models.domain import RetrievalResult


SYSTEM_PROMPT = """You are a helpful knowledge assistant. Answer the user's question using ONLY the provided context below.

Rules:
- Use only the information from the provided sources to answer.
- Do NOT use your general knowledge to fill in missing facts.
- If the context does not contain enough information to answer the question, say: "I don't have enough information in the indexed documents to answer that reliably."
- Be concise unless the user asks for detail.
- Cite sources ONLY using the labels provided (e.g. [S1], [S2]). Never invent a source label that was not provided.
- Do not invent document names or page numbers — use only the labels."""


def build_context(chunks: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a structured context block for the LLM.

    Uses stable [S1], [S2], ... labels so the LLM cites them by label
    instead of freely inventing document/page references.
    """
    sections = []

    for i, chunk in enumerate(chunks, start=1):
        page_info = f"Page: {chunk.page_number}" if chunk.page_number else "Page: N/A"
        section = (
            f"[S{i}]\n"
            f"Document: {chunk.document_name}\n"
            f"{page_info}\n"
            f"Content:\n{chunk.text}"
        )
        sections.append(section)

    return "\n\n".join(sections)


def build_user_prompt(question: str, context: str) -> str:
    """Build the user message containing the context and question."""
    return f"""Context:
{context}

Question: {question}"""
