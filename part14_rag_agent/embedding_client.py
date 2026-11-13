"""
Embedding generation via Ollama's /api/embed endpoint.

Identical in approach to part11_grounded_rag/embedder.py and
part12_rag_evaluation/embedder.py - same endpoint, same dimension
check. Kept as its own small module (rather than imported) to match
this repo's existing convention of each part owning its own thin
infra files.
"""

import requests

from config import OLLAMA_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingError(Exception):
    """Raised when Ollama is unreachable or returns something unusable."""


def get_embedding(text: str):
    url = f"{OLLAMA_URL}/api/embed"

    try:
        response = requests.post(
            url,
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        raise EmbeddingError(f"Could not reach Ollama at {url}: {error}") from error

    try:
        embedding = data["embeddings"][0]
    except (KeyError, IndexError) as error:
        raise EmbeddingError(
            f"Unexpected embedding response shape: {data}"
        ) from error

    if len(embedding) != EMBEDDING_DIMENSION:
        raise EmbeddingError(
            f"Expected {EMBEDDING_DIMENSION} dimensions, received {len(embedding)}"
        )

    return embedding
