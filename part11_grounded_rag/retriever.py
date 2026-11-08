from pgvector import Vector

from db import get_connection
from embedder import get_embedding
from config import TOP_K, MIN_SIMILARITY


def retrieve_chunks(query, top_k=TOP_K):
    query_embedding = get_embedding(query)

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    chunk_key,
                    document_id,
                    source,
                    page_number,
                    chunk_index,
                    content,
                    metadata,
                    1 - (embedding <=> %s) AS similarity
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    Vector(query_embedding),
                    Vector(query_embedding),
                    top_k
                )
            )

            rows = cur.fetchall()

        results = []

        for row in rows:
            (
                chunk_id,
                chunk_key,
                document_id,
                source,
                page_number,
                chunk_index,
                content,
                metadata,
                similarity
            ) = row

            similarity = float(similarity)

            if similarity < MIN_SIMILARITY:
                continue

            results.append({
                "id": chunk_id,
                "chunk_key": chunk_key,
                "document_id": document_id,
                "source": source,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "content": content,
                "metadata": metadata or {},
                "similarity": similarity
            })

        return results

    finally:
        conn.close()