import psycopg
from pgvector.psycopg import register_vector

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