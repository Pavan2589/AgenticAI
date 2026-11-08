from retriever import retrieve_chunks
from generator import generate_answer

from config import MAX_CONTEXT_CHARS


def build_context(chunks):
    context_parts = []
    sources = {}
    total_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        source_id = f"S{index}"

        content = (
            f"[{source_id}]\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Content: {chunk['content']}\n"
        )

        if total_chars + len(content) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(content)

        sources[source_id] = {
            "source_id": source_id,
            "chunk_id": chunk["id"],
            "chunk_key": chunk["chunk_key"],
            "document_id": chunk["document_id"],
            "source": chunk["source"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "similarity": chunk["similarity"]
        }

        total_chars += len(content)

    return "\n".join(context_parts), sources


def answer_question(question):
    print("\nRetrieving relevant chunks...")

    chunks = retrieve_chunks(question)

    if not chunks:
        return {
            "answer": "The available documents do not provide enough evidence.",
            "insufficient_evidence": True,
            "citations": [],
            "sources": []
        }

    context, sources = build_context(chunks)

    if not sources:
        return {
            "answer": "The available documents do not provide enough evidence.",
            "insufficient_evidence": True,
            "citations": [],
            "sources": []
        }

    print(f"Retrieved {len(sources)} chunks.")
    print("Generating answer...")

    result = generate_answer(question, context)

    valid_citations = list(dict.fromkeys(
        citation
        for citation in result["citations"]
        if isinstance(citation, str) and citation in sources
    ))

    if result["insufficient_evidence"]:
        valid_citations = []

    elif not valid_citations:
        result["answer"] = (
            "I couldn't verify the sources supporting the generated answer."
        )
        result["insufficient_evidence"] = True

    result["citations"] = valid_citations
    result["sources"] = [
        sources[citation] for citation in valid_citations
    ]

    return result