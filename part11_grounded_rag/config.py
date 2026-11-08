import os
from dotenv import load_dotenv

load_dotenv()

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "agentic_ai")
DB_USER = os.getenv("DB_USER", "pavan")
DB_PASSWORD = os.getenv("DB_PASSWORD", "day9password")

# Ollama
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3:4b"
)

# RAG
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(
    os.getenv("MIN_SIMILARITY", "0.25")
)
MAX_CONTEXT_CHARS = int(
    os.getenv("MAX_CONTEXT_CHARS", "12000")
)

EMBEDDING_DIMENSION = 768