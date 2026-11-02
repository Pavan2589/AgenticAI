# Day 9 — Embeddings and Vector Search

A semantic document retrieval system built using **Ollama, PostgreSQL, and pgvector**. This project converts document chunks into vector embeddings, stores them in a PostgreSQL database, and retrieves relevant information using semantic similarity.

## Overview

Traditional keyword-based search relies on matching exact words. Vector search allows us to retrieve information based on semantic meaning, even when the query and document use different words.

In this project, we use the `nomic-embed-text` embedding model through Ollama to generate 768-dimensional embeddings and pgvector to perform similarity searches.

## Architecture

```text
Day 8 — Document Ingestion
          |
          v
  Semantic Chunking
          |
          v
   Chunk JSON File
          |
          v
   Ollama Embeddings
   (nomic-embed-text)
          |
          v
 PostgreSQL + pgvector
          |
          v
      User Query
          |
          v
    Query Embedding
          |
          v
   Cosine Similarity
          |
          v
   Top-K Relevant Chunks
```

## Tech Stack

- **Python** — Application logic
- **Ollama** — Local embedding generation
- **nomic-embed-text** — Embedding model (768 dimensions)
- **PostgreSQL** — Relational database
- **pgvector** — Vector storage and similarity search
- **Docker** — PostgreSQL container
- **Psycopg** — PostgreSQL connectivity
- **python-dotenv** — Environment configuration

## Project Structure

```text
day9_embeddings_vector_search/
│
├── docker-compose.yml
├── .env
├── requirements.txt
├── README.md
│
├── db.py
├── embedder.py
├── ingest.py
├── search.py
├── cleanup.py
└── main.py
```

## Concepts Covered

### 1. Embeddings

Embeddings are numerical representations of text that capture semantic information.

Text is converted into a vector of 768 numerical values using `nomic-embed-text`.

### 2. Vector Similarity

We use cosine similarity to measure how close two embedding vectors are in terms of direction.

The system uses pgvector's cosine distance operator:

```sql
embedding <=> query_embedding
```

Cosine similarity is calculated as:

```text
similarity = 1 - cosine_distance
```

### 3. Vector Database — pgvector

pgvector is a PostgreSQL extension that enables vector storage and similarity search.

Instead of introducing a separate vector database, we use PostgreSQL with pgvector to store both document metadata and embeddings.

### 4. Top-K Retrieval

For every user query, the system retrieves the K most similar document chunks.

The current application retrieves the top 5 results by default.

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker
- Ollama

### 1. Install the Ollama embedding model

```bash
ollama pull nomic-embed-text
```

Ensure Ollama is running.

### 2. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

The PostgreSQL container uses host port `5433`.

### 3. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file:

```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=agentic_ai
DB_USER=pavan
DB_PASSWORD=your_password

OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text

CHUNK_FILE=../day8_document_ingestion/output/semantic_chunks.json
```

Use your own database credentials. Never commit `.env` to GitHub.

### 6. Run the application

```bash
python main.py
```

Choose:

- `1` — Ingest document chunks and generate embeddings.
- `2` — Search the chunks already stored in PostgreSQL.

## Example Queries

Try searching for:

```text
What is the role of blockchain in vehicular edge computing?

How does blockchain provide security and trust?

Explain the consensus mechanism used in the proposed framework.

How are blockchain and quantum key distribution integrated?

What is the role of blockchain in resource allocation?
```

The system returns the top 5 relevant chunks, their similarity scores, source documents, and page numbers.

## Database Management

To remove previously ingested chunks:

```bash
python cleanup.py
```

This clears the records from the `document_chunks` table without dropping the table or the pgvector extension.

## Key Learnings

- Understanding embedding models and vector dimensions.
- Generating embeddings using a locally running LLM ecosystem.
- Understanding cosine similarity and vector distance.
- Storing embeddings in PostgreSQL using pgvector.
- Performing semantic search using SQL.
- Connecting document ingestion and chunking with vector retrieval.
- Understanding the limitations of Top-K retrieval and semantic similarity.

## Current Implementation

This project implements a **vector retrieval pipeline**, not a complete RAG application.

The next step is to connect the retrieved chunks to a local LLM and generate context-grounded answers using Retrieval-Augmented Generation (RAG).

---

**Part of my 4-Week Agentic AI Learning Journey.**
