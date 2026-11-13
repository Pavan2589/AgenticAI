"""
Postgres/pgvector connection layer.

Adapted from part12_rag_evaluation/db.py (itself adapted from Part 11).
Same table (`document_chunks`), same connection parameters, same
cosine-distance query shape. The only addition is an explicit
`document_id` scope filter in `search_chunks`, so Part 14's RAG tool
can never surface the pre-existing research-paper chunks as policy
evidence (see config.KNOWLEDGE_BASE_DOCUMENT_IDS).
"""

import psycopg
from pgvector.psycopg import register_vector
from pgvector import Vector

from config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)


def get_connection():
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    register_vector(conn)
    return conn


def get_chunk_count(document_ids=None):
    """Total chunk count, optionally scoped to a set of document_ids."""

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            if document_ids is not None:
                cur.execute(
                    "SELECT COUNT(*) FROM document_chunks "
                    "WHERE document_id = ANY(%s)",
                    (list(document_ids),),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM document_chunks")

            return cur.fetchone()[0]

    finally:
        conn.close()


def table_exists():
    """Used by ingest.py to verify the Part 9 schema before inserting."""

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'document_chunks'
                )
                """
            )
            return cur.fetchone()[0]

    finally:
        conn.close()


def verify_schema():
    """Fail clearly if the shared Parts 9-12 table is incompatible."""
    required = {
        "id", "chunk_key", "document_id", "source", "page_number",
        "chunk_index", "content", "embedding", "metadata",
    }
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'document_chunks'"
            )
            present = {row[0] for row in cur.fetchall()}
        missing = required - present
        if missing:
            raise RuntimeError(
                "document_chunks schema is missing required columns: "
                + ", ".join(sorted(missing))
            )
    finally:
        conn.close()


def insert_chunk(chunk, embedding):
    """
    Insert a single chunk, deduplicating on chunk_key exactly like
    Part 9's ingest.py. Returns True if a new row was inserted,
    False if it already existed (ON CONFLICT DO NOTHING -> no row
    returned by RETURNING).
    """

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            from psycopg.types.json import Jsonb

            cur.execute(
                """
                INSERT INTO document_chunks (
                    chunk_key, document_id, source, page_number,
                    chunk_index, content, embedding, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_key) DO NOTHING
                RETURNING id
                """,
                (
                    chunk["chunk_key"],
                    chunk["document_id"],
                    chunk["source"],
                    chunk["page_number"],
                    chunk["chunk_index"],
                    chunk["content"],
                    Vector(embedding),
                    Jsonb(chunk["metadata"]),
                ),
            )

            inserted = cur.fetchone() is not None
            conn.commit()
            return inserted

    finally:
        conn.close()


def search_chunks(query_embedding, top_k=5, document_ids=None):
    """
    Unfiltered-by-similarity top-k search (filtering by MIN_SIMILARITY
    happens in retriever.py, same split as Part 11/12). When
    document_ids is given, only chunks whose document_id is in that
    set are considered - this is what keeps the unrelated research
    corpus out of policy answers.
    """

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            if document_ids is not None:
                cur.execute(
                    """
                    SELECT
                        id, chunk_key, document_id, source, page_number,
                        chunk_index, content, metadata,
                        embedding <=> %s AS distance
                    FROM document_chunks
                    WHERE embedding IS NOT NULL
                      AND document_id = ANY(%s)
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (
                        Vector(query_embedding),
                        list(document_ids),
                        Vector(query_embedding),
                        top_k,
                    ),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        id, chunk_key, document_id, source, page_number,
                        chunk_index, content, metadata,
                        embedding <=> %s AS distance
                    FROM document_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (Vector(query_embedding), Vector(query_embedding), top_k),
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
                    "similarity": 1 - float(row[8]),
                })

            return results

    finally:
        conn.close()
