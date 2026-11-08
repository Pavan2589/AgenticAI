import requests

from config import (
    OLLAMA_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION
)


def get_embedding(text):
    url = f"{OLLAMA_URL}/api/embed"

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    embedding = data["embeddings"][0]

    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION} dimensions, "
            f"received {len(embedding)}"
        )

    return embedding