
import os
import json
import hashlib

from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from db import get_connection, create_table
from embedder import generate_embeddings

load_dotenv()

CHUNK_FILE = os.getenv(
    "CHUNK_FILE",
    "../day8_document_ingestion/output/semantic_chunks.json"
)

BATCH_SIZE = 8


def load_chunks():
    with open(CHUNK_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Support either a list of chunks or a dictionary containing a list.
    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict):
        chunks = next(
            (
                data[key]
                for key in ("chunks", "data", "results")
                if isinstance(data.get(key), list)
            ),
            None
        )

        if chunks is None:
            raise ValueError(
                "Could not find a chunk list in the JSON file."
            )
    else:
        raise ValueError("Unsupported JSON structure.")

    return chunks


def get_chunk_details(chunk, index):
    """
    Extract common chunk fields.
    Adjust these keys if your Day 8 JSON uses different names.
    """

    if isinstance(chunk, str):
        content = chunk
        metadata = {}
    elif isinstance(chunk, dict):
        content = (
            chunk.get("content")
            or chunk.get("text")
            or chunk.get("chunk")
            or chunk.get("page_content")
        )

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Chunk {index} has no valid text field. "
                f"Available keys: {list(chunk.keys())}"
            )

        metadata = chunk.get("metadata") or {}
    else:
        raise ValueError(f"Unsupported chunk at index {index}")

    if not isinstance(metadata, dict):
        metadata = {"original_metadata": metadata}

    source = (
        chunk.get("source")
        or chunk.get("source_file")
        or metadata.get("source")
        or metadata.get("source_file")
        if isinstance(chunk, dict)
        else None
    )

    page_number = (
        chunk.get("page_number", metadata.get("page_number"))
        if isinstance(chunk, dict)
        else None
    )

    document_id = (
        chunk.get("document_id", metadata.get("document_id"))
        if isinstance(chunk, dict)
        else None
    )

    chunk_index = (
        chunk.get("chunk_index", index)
        if isinstance(chunk, dict)
        else index
    )

    # Create a stable unique key for each chunk.
    identity = {
        "source": source,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "content": content
    }

    chunk_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "chunk_key": chunk_key,
        "document_id": str(document_id) if document_id is not None else None,
        "source": str(source) if source is not None else None,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "content": content,
        "metadata": metadata
    }


def ingest_documents():
    create_table()

    raw_chunks = load_chunks()

    if not raw_chunks:
        print("No chunks found.")
        return

    print(f"Loaded {len(raw_chunks)} chunks.")

    chunks = [
        get_chunk_details(chunk, index)
        for index, chunk in enumerate(raw_chunks)
    ]

    total_inserted = 0

    with get_connection() as conn:
        with conn.cursor() as cur:

            for start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[start:start + BATCH_SIZE]

                texts = [chunk["content"] for chunk in batch]

                embeddings = generate_embeddings(texts)

                for chunk, embedding in zip(batch, embeddings):
                    cur.execute(
                        """
                        INSERT INTO document_chunks (
                            chunk_key,
                            document_id,
                            source,
                            page_number,
                            chunk_index,
                            content,
                            embedding,
                            metadata
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (chunk_key) DO NOTHING
                        """,
                        (
                            chunk["chunk_key"],
                            chunk["document_id"],
                            chunk["source"],
                            chunk["page_number"],
                            chunk["chunk_index"],
                            chunk["content"],
                            embedding,
                            Jsonb(chunk["metadata"])
                        )
                    )

                conn.commit()

                total_inserted += len(batch)

                print(
                    f"Processed {min(start + BATCH_SIZE, len(chunks))}"
                    f"/{len(chunks)} chunks"
                )

    print("Ingestion completed!")
    print(f"Processed chunks: {total_inserted}")


if __name__ == "__main__":
    ingest_documents()
