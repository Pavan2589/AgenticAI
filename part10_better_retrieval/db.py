
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Establish a connection to PostgreSQL."""

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def fetch_all_chunks():
    """Fetch all document chunks from PostgreSQL."""

    conn = get_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    id,
                    chunk_key,
                    document_id,
                    source,
                    page_number,
                    chunk_index,
                    content,
                    metadata
                FROM document_chunks
                ORDER BY id;
            """)

            return cur.fetchall()

    finally:
        conn.close()


def get_chunk_count():
    """Return the total number of stored chunks."""

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            return cur.fetchone()[0]

    finally:
        conn.close()
