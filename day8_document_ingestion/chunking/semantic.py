import re
import requests
import math


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def get_embeddings(sentences: list[str]) -> list[list[float]]:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBED_MODEL,
            "input": sentences
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.55,
    max_chunk_size: int = 1000
) -> list[str]:

    sentences = split_sentences(text)

    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    embeddings = get_embeddings(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            embeddings[i - 1],
            embeddings[i]
        )

        candidate = " ".join(current_chunk + [sentences[i]])

        if (
            similarity < similarity_threshold
            or len(candidate) > max_chunk_size
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks