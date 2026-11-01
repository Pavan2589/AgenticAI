import re


def clean_text(text: str) -> str:
    # Normalize Windows and Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unnecessary spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    # Avoid excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()