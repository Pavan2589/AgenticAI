from rag import answer_question
from db import get_chunk_count


def display_result(result):
    print("\n" + "=" * 65)
    print("ANSWER")
    print("=" * 65)

    print(result["answer"])

    if result["insufficient_evidence"]:
        print("\n[INSUFFICIENT EVIDENCE]")
    else:
        print("\nCITATIONS")

        for citation in result["citations"]:
            print(f"[{citation}]")

    if result["sources"]:
        print("\nSOURCE DETAILS")

        for source in result["sources"]:
            print("-" * 50)

            print(f"Source ID: {source['source_id']}")
            print(f"Document: {source['source']}")
            print(f"Page: {source['page_number']}")
            print(f"Chunk ID: {source['chunk_id']}")
            print(
                f"Similarity: {source['similarity']:.4f}"
            )

    print("=" * 65)


def main():
    print("\nGROUNDed RAG ASSISTANT")
    print("-" * 40)

    try:
        count = get_chunk_count()
        print(f"Available document chunks: {count}")

    except Exception as error:
        print(f"Database connection failed: {error}")
        return

    if count == 0:
        print("No document chunks found.")
        return

    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Exiting RAG assistant.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = answer_question(question)
            display_result(result)

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()