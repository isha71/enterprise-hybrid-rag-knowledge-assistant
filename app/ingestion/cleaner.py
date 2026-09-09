import re


def clean_text(text: str) -> str:
    """Lightweight text cleaning that preserves meaningful content."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of whitespace (spaces/tabs) within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace from the entire text
    text = text.strip()

    return text
