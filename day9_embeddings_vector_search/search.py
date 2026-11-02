
from db import get_connection
from embedder import generate_embeddings
from pgvector import Vector


def semantic_search(query, top_k=5):
    query_embedding = generate_embeddings([query])[0]

    # Explicitly convert the Python list into a pgvector Vector.
    query_vector = Vector(query_embedding)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    source,
                    page_number,
                    content,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, top_k)
            )

            results = cur.fetchall()

    return [
        {
            "id": row[0],
            "source": row[1],
            "page_number": row[2],
            "content": row[3],
            "similarity": float(row[4])
        }
        for row in results
    ]


if __name__ == "__main__":
    query = input("Enter your search query: ")

    results = semantic_search(query)

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("-" * 50)
        print("Similarity:", round(result["similarity"], 4))
        print("Source:", result["source"])
        print("Page:", result["page_number"])
        print("Content:", result["content"])
