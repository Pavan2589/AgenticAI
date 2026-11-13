import json
from langchain_core.tools import tool
from config import MAX_CONTEXT_CHARS
from retriever import retrieve_chunks, STATUS_RELEVANT, STATUS_ERROR, STATUS_NO_RESULTS


def _source_record(chunk, source_id):
    return {
        "source_id": source_id, "chunk_id": chunk["id"],
        "chunk_key": chunk["chunk_key"], "document_id": chunk["document_id"],
        "source": chunk["source"], "page_number": chunk.get("page_number"),
        "similarity": chunk["similarity"],
        "document_version": (chunk.get("metadata") or {}).get("document_version"),
        "effective_date": (chunk.get("metadata") or {}).get("effective_date"),
    }


def format_retrieval(result):
    if result.status == STATUS_ERROR:
        return {"status": "error", "message": "Knowledge search is temporarily unavailable.",
                "diagnostic": result.error_message, "sources": [], "evidence": []}
    if result.status == STATUS_NO_RESULTS:
        return {"status": "no_results", "message": "No documents are available in the knowledge base.",
                "sources": [], "evidence": []}
    if result.status != STATUS_RELEVANT:
        return {"status": "low_relevance", "message": "No sufficiently relevant document evidence was found.",
                "best_similarity": result.best_similarity, "sources": [], "evidence": []}

    sources, evidence, used = [], [], 0
    for n, chunk in enumerate(result.chunks, 1):
        record = _source_record(chunk, f"S{n}")
        entry = {**record, "content": chunk["content"]}
        size = len(json.dumps(entry, ensure_ascii=False))
        if used + size > MAX_CONTEXT_CHARS:
            break
        sources.append(record)
        evidence.append(entry)
        used += size
    return {"status": "relevant", "sources": sources, "evidence": evidence}


@tool
def search_knowledge_base(query: str) -> str:
    """Search internal policy, product, and technical documents for evidence."""
    result = retrieve_chunks(query)
    return json.dumps(format_retrieval(result), ensure_ascii=False)
