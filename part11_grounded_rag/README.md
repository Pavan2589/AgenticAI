# Day 11 — Grounded RAG Assistant

## Overview

A Retrieval-Augmented Generation (RAG) assistant that answers
questions using retrieved document chunks and provides source
references.

## Architecture

User Question
      |
      v
Query Embedding (Nomic Embed Text)
      |
      v
PostgreSQL + pgvector
      |
      v
Top-K Relevant Chunks
      |
      v
Context Construction + Source IDs
      |
      v
Qwen3:4b (Ollama)
      |
      v
Grounded Answer + Citations
      |
      v
Citation Validation

## Tech Stack

- Python
- Ollama
- Qwen3:4b
- Nomic Embed Text
- PostgreSQL
- pgvector

## Features

- Semantic retrieval from existing document chunks
- Context injection into the LLM prompt
- Source ID generation
- Citation ID validation
- Insufficient-evidence handling
- Structured JSON responses

## Run

Activate the virtual environment:

    source ../venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Start the application:

    python main.py

## Environment

Configure PostgreSQL and Ollama settings in `.env`.

Do not commit `.env` because it contains credentials.

## Limitations

- Citation validation checks source ID existence, not
  semantic correctness.
- Similarity threshold requires empirical tuning.
- LLM-generated answers may still contain unsupported claims.
- Evaluation requires manually checking answers against
  retrieved evidence.