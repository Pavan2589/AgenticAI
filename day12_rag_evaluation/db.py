
import psycopg
from pgvector.psycopg import register_vector
from pgvector import Vector

from config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD
)


def get_connection():
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    register_vector(conn)
    return conn


def get_chunk_count():
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM document_chunks"
            )
            return cur.fetchone()[0]

    finally:
        conn.close()



def search_chunks(query_embedding, top_k=5):
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
                    embedding <=> %s AS distance
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
                results.append({
                    "id": row[0],
                    "chunk_key": row[1],
                    "document_id": row[2],
                    "source": row[3],
                    "page_number": row[4],
                    "chunk_index": row[5],
                    "content": row[6],
                    "metadata": row[7] or {},
                    "distance": float(row[8]),
                    "similarity": 1 - float(row[8])
                })

            return results

    finally:
        conn.close()
