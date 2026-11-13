"""
Central configuration for Part 14.

Reuses the exact connection/embedding conventions established in
Part 11 (part11_grounded_rag/config.py) and Part 12
(part12_rag_evaluation/config.py): same Postgres container
(day9_pgvector on port 5433), same embedding model, same default
similarity threshold. This keeps Part 14 compatible with the data
already sitting in `document_chunks` instead of inventing a new
convention.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Database (day9_pgvector container, shared with Parts 9-12)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "agentic_ai")
DB_USER = os.getenv("DB_USER", "pavan")
DB_PASSWORD = os.getenv("DB_PASSWORD", "day9password")

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
EMBEDDING_DIMENSION = 768

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# Calibrated for THIS corpus (see README "Calibrating the relevance
# threshold"): on the Part 14 documents, genuinely on-topic queries
# scored 0.54-0.81 cosine similarity while clearly off-topic queries
# ("capital of Japan", "125 * 48") scored 0.35-0.45. 0.48 sits in the
# gap. Part 11/12 used 0.25 for a different, larger corpus - do not
# assume either value transfers to a new document set without
# re-running evaluation/evaluate_retrieval.py.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.67"))

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

# Documents whose chunks are eligible to answer policy/product questions.
# This scopes retrieval so the 1,807 pre-existing research-paper chunks
# (ingested in Parts 9-12) never get presented as policy evidence, even
# if they happen to score above MIN_SIMILARITY for some query.
KNOWLEDGE_BASE_DOCUMENT_IDS = [
    "employee_leave_policy",
    "employee_leave_policy_old",
    "company_refund_policy",
    "product_documentation",
    "technical_guide",
]
