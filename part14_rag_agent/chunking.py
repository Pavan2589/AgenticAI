"""
Recursive text chunking.

Same recursive-split-then-merge-with-overlap strategy as
part08_document_ingestion/chunking/recursive.py, reimplemented here
(not imported) because Part 8's version is a standalone script-style
module, not a stable package Part 14 should depend on at runtime.
"""

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def recursive_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    if not text.strip():
        return []

    separators = ("\n\n", "\n", ". ", " ")
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            # Prefer natural boundaries without discarding their characters.
            candidates = [text.rfind(sep, start + 1, end + 1) for sep in separators]
            boundary = max(candidates)
            if boundary > start:
                matched = next(sep for sep in separators if text.startswith(sep, boundary))
                end = min(boundary + len(matched), text_length)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        start = max(start + 1, end - overlap)
    return chunks
