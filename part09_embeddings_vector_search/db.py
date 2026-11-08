
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def get_connection():
    conn = psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5433"),
        dbname=os.getenv("DB_NAME", "agentic_ai"),
        user=os.getenv("DB_USER", "pavan"),
        password=os.getenv("DB_PASSWORD")
    )

    # Make sure the pgvector extension exists before registering its type.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    return conn


def create_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                chunk_key TEXT UNIQUE NOT NULL,
                document_id TEXT,
                source TEXT,
                page_number INTEGER,
                chunk_index INTEGER,
                content TEXT NOT NULL,
                embedding VECTOR(768),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        conn.commit()

    print("Document chunks table created successfully!")


if __name__ == "__main__":
    create_table()
