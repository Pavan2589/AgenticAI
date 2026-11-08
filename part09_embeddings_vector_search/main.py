
from db import create_table, get_connection
from ingest import ingest_documents
from search import semantic_search


def show_database_stats():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            count = cur.fetchone()[0]

    print(f"\nTotal chunks in database: {count}")


def main():
    print("=" * 55)
    print("       DAY 9 - VECTOR RETRIEVAL SYSTEM")
    print("=" * 55)

    create_table()

    print("\n1. Ingest documents")
    print("2. Search existing documents")

    choice = input("\nChoose an option (1/2): ").strip()

    if choice == "1":
        ingest_documents()

    elif choice != "2":
        print("Invalid option.")
        return

    show_database_stats()

    print("\nSemantic Search is ready!")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ").strip()

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not query:
            continue

        results = semantic_search(query, top_k=5)

        if not results:
            print("No matching documents found.")
            continue

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")
            print("-" * 55)
            print("Similarity:", round(result["similarity"], 4))
            print("Source:", result["source"])
            print("Page:", result["page_number"])
            print("\n", result["content"])


if __name__ == "__main__":
    main()
