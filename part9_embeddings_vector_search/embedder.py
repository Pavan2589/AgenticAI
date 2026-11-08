
import os
import requests

from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)


def generate_embeddings(texts):
    """
    Accepts a list of strings and returns
    a list of 768-dimensional embeddings.
    """

    if not texts:
        return []

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=300
    )

    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    if len(embeddings) != len(texts):
        raise ValueError("Embedding count does not match input count")

    for embedding in embeddings:
        if len(embedding) != 768:
            raise ValueError(
                f"Expected 768 dimensions, received {len(embedding)}"
            )

    return embeddings


if __name__ == "__main__":
    sample_text = "Vector databases store numerical representations of text."

    result = generate_embeddings([sample_text])

    print("Embedding generated successfully!")
    print("Dimensions:", len(result[0]))
    print("First 10 values:", result[0][:10])
