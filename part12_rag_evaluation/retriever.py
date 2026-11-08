
from embedder import get_embedding
from db import search_chunks
from config import MIN_SIMILARITY


def retrieve_chunks(query, top_k=5):
    """
    Retrieve the top-k semantically similar chunks.

    Applies the same minimum similarity threshold
    used by the Day 11 RAG implementation.
    """

    query_embedding = get_embedding(query)

    chunks = search_chunks(
        query_embedding=query_embedding,
        top_k=top_k
    )

    results = []

    for chunk in chunks:
        if chunk["similarity"] < MIN_SIMILARITY:
            continue

        results.append(chunk)

    return results
