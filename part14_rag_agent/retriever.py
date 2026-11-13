"""
Retrieval with explicit relevance classification.

Part 11/12's retriever.py silently dropped chunks below
MIN_SIMILARITY. Part 14 needs to tell the difference between "the
knowledge base has nothing on this topic" (low_relevance - chunks
exist but aren't good enough) and "the knowledge base is literally
empty" (no_results), plus a distinct "error" status for infra
failures, so the RAG tool and the evaluation suite can react
differently to each (see rag_tool.py and evaluation/test_cases.json
scenario G).
"""

from dataclasses import dataclass, field
import re

from embedding_client import get_embedding, EmbeddingError
from db import search_chunks
from config import TOP_K, MIN_SIMILARITY, KNOWLEDGE_BASE_DOCUMENT_IDS

STATUS_RELEVANT = "relevant"
STATUS_LOW_RELEVANCE = "low_relevance"
STATUS_NO_RESULTS = "no_results"
STATUS_ERROR = "error"

_GENERIC_QUERY_TERMS = {
    "what", "which", "who", "when", "where", "why", "how", "many", "much",
    "company", "companys", "current", "policy", "policies", "employee", "employees",
    "annual", "paid", "leave", "leaves", "day", "days", "allowance", "entitlement",
    "remain", "remaining", "receive", "receives", "received", "use", "uses", "used",
    "already", "under", "about", "does", "did", "can", "could", "would", "should",
    "the", "our", "their", "this", "that", "with", "from", "into", "for", "and", "are",
    "is", "be", "if", "an", "a", "of", "to", "in", "on", "at", "by", "tell", "please",
    "request", "requests", "say", "says", "get", "gets", "provide", "provides",
}


def _normalize_term(term: str) -> str:
    if term == "leaves":
        return "leave"
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith("ed") and len(term) > 5:
        return term[:-2]
    if term.endswith("s") and len(term) > 4:
        return term[:-1]
    return term


def _has_topic_overlap(query: str, chunks: list[dict]) -> bool:
    """Reject high-similarity chunks that do not share query-specific terms."""
    terms = {
        _normalize_term(word.lower())
        for word in re.findall(r"[A-Za-z]+", query)
        if len(word) > 3 and word.lower() not in _GENERIC_QUERY_TERMS
    }
    if not terms:
        return True
    content_terms = {
        _normalize_term(word.lower())
        for chunk in chunks
        for word in re.findall(r"[A-Za-z]+", chunk["content"])
    }
    return len(terms & content_terms) / len(terms) >= 0.5


@dataclass
class RetrievalResult:
    status: str
    chunks: list = field(default_factory=list)
    best_similarity: float | None = None
    error_message: str | None = None


def retrieve_chunks(
    query: str,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    document_ids=KNOWLEDGE_BASE_DOCUMENT_IDS,
) -> RetrievalResult:
    try:
        query_embedding = get_embedding(query)
    except Exception as error:
        return RetrievalResult(status=STATUS_ERROR, error_message=str(error))

    try:
        # Fetch extra candidates because corrected chunker v2 rows may be
        # interleaved with the preserved legacy rows for the same documents.
        candidates = search_chunks(
            query_embedding, top_k=max(top_k * 3, top_k), document_ids=document_ids
        )
    except Exception as error:  # psycopg errors, connection failures, etc.
        return RetrievalResult(status=STATUS_ERROR, error_message=str(error))

    if not candidates:
        return RetrievalResult(status=STATUS_NO_RESULTS)

    best_similarity = max(c["similarity"] for c in candidates)

    # Prefer corrected chunker output if both preserved legacy and v2 rows
    # exist for a document. No legacy rows are deleted or modified.
    versioned_documents = {
        c["document_id"] for c in candidates
        if (c.get("metadata") or {}).get("chunker_version") == 2
    }
    candidates = [
        c for c in candidates
        if c.get("document_id") not in versioned_documents
        or (c.get("metadata") or {}).get("chunker_version") == 2
    ]

    # Policy status is an authority signal in addition to cosine similarity.
    # If a current version appears among candidates, suppress superseded
    # versions so near-duplicate old policies cannot win by ranking first.
    current_policy_ids = {
        c["document_id"] for c in candidates
        if (c.get("metadata") or {}).get("policy_status") == "current"
    }
    candidates = [
        c for c in candidates
        if not (
            (c.get("metadata") or {}).get("policy_status") == "superseded"
            and c.get("document_id", "").endswith("_old")
            and any(
                current_id == c.get("document_id", "").removesuffix("_old")
                for current_id in current_policy_ids
            )
        )
    ]
    candidates = candidates[:top_k]
    best_similarity = max((c["similarity"] for c in candidates), default=best_similarity)
    relevant = [c for c in candidates if c["similarity"] >= min_similarity]

    if relevant and not _has_topic_overlap(query, relevant):
        return RetrievalResult(
            status=STATUS_LOW_RELEVANCE,
            chunks=candidates,
            best_similarity=best_similarity,
        )

    if not relevant:
        return RetrievalResult(
            status=STATUS_LOW_RELEVANCE,
            chunks=candidates,
            best_similarity=best_similarity,
        )

    return RetrievalResult(
        status=STATUS_RELEVANT,
        chunks=relevant,
        best_similarity=best_similarity,
    )
