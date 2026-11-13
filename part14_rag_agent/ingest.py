"""
Ingestion pipeline: documents/ -> chunks -> embeddings -> document_chunks.

Reuses Part 9's dedup strategy (ingest.py): a chunk_key hashed from
(source, page_number, chunk_index, content) is inserted with
ON CONFLICT DO NOTHING, so re-running `python ingest.py` after the
documents are already loaded is a safe no-op instead of creating
duplicates.

Each document gets explicit policy metadata (policy_status,
effective_date, document_version) here, rather than parsed out of the
document body - in a real system this metadata would come from a
document management system, not be scraped from free text, and
parsing it from text would be fragile. This metadata is what lets the
agent prefer the current leave policy over the superseded one without
relying on embedding similarity alone (see retriever.py and
prompts.py).
"""

import hashlib
import json
import sys
from pathlib import Path

from chunking import recursive_chunk
from document_loader import load_document
from embedding_client import get_embedding, EmbeddingError
from db import insert_chunk, table_exists, verify_schema

DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Explicit per-document metadata. document_id matches the filename
# stem, which is also what config.KNOWLEDGE_BASE_DOCUMENT_IDS scopes
# retrieval to.
DOCUMENT_METADATA = {
    "employee_leave_policy": {
        "policy_status": "current",
        "effective_date": "2026-01-01",
        "document_version": "2.0",
        "chunker_version": 2,
    },
    "employee_leave_policy_old": {
        "policy_status": "superseded",
        "effective_date": "2023-01-01",
        "document_version": "1.0",
        "chunker_version": 2,
    },
    "company_refund_policy": {
        "policy_status": "current",
        "effective_date": "2025-06-01",
        "document_version": "1.0",
        "chunker_version": 2,
    },
    "product_documentation": {
        "policy_status": "current",
        "effective_date": "2025-09-15",
        "document_version": "3.1",
        "chunker_version": 2,
    },
    "technical_guide": {
        "policy_status": "current",
        "effective_date": "2025-09-15",
        "document_version": "1.4",
        "chunker_version": 2,
    },
}


def make_chunk_key(source, page_number, chunk_index, content):
    identity = {
        "source": source,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "content": content,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()


def ingest_documents():
    if not table_exists():
        print(
            "ERROR: 'document_chunks' table does not exist. "
            "Run part09_embeddings_vector_search/db.py's create_table() "
            "first (it already has been, for the existing 1,807 chunks)."
        )
        sys.exit(1)

    verify_schema()

    files = sorted(DOCUMENTS_DIR.glob("*.txt")) + sorted(
        DOCUMENTS_DIR.glob("*.md")
    ) + sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not files:
        print(f"No documents found in {DOCUMENTS_DIR}")
        return

    total_inserted = 0
    total_skipped = 0

    for file_path in files:
        document_id = file_path.stem
        metadata = DOCUMENT_METADATA.get(document_id, {})

        if document_id not in DOCUMENT_METADATA:
            print(
                f"WARNING: no metadata entry for '{document_id}', "
                f"ingesting with empty policy metadata."
            )

        pages = load_document(str(file_path))
        print(f"\n{file_path.name}: {len(pages)} page(s)/section(s)")

        chunk_index = 0

        for page in pages:
            chunks = recursive_chunk(page["text"])

            for chunk_text in chunks:
                chunk_key = make_chunk_key(
                    page["source"], page["page_number"], chunk_index, chunk_text
                )

                try:
                    embedding = get_embedding(chunk_text)
                except EmbeddingError as error:
                    print(f"  Embedding failed for chunk {chunk_index}: {error}")
                    sys.exit(1)

                inserted = insert_chunk(
                    {
                        "chunk_key": chunk_key,
                        "document_id": document_id,
                        "source": page["source"],
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                        "content": chunk_text,
                        "metadata": metadata,
                    },
                    embedding,
                )

                if inserted:
                    total_inserted += 1
                else:
                    total_skipped += 1

                chunk_index += 1

        print(f"  Chunks processed: {chunk_index}")

    print("\nIngestion complete.")
    print(f"New chunks inserted: {total_inserted}")
    print(f"Already present (skipped): {total_skipped}")


if __name__ == "__main__":
    ingest_documents()
