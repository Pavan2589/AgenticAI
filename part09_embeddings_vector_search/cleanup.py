
from db import get_connection


def cleanup_database():
    confirm = input(
        "Are you sure you want to delete ALL ingested chunks? (yes/no): "
    )

    if confirm.lower() != "yes":
        print("Cleanup cancelled.")
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks")

            deleted_count = cur.rowcount

        conn.commit()

    print(f"Successfully deleted {deleted_count} chunks.")
    print("Your document_chunks table is now empty.")


if __name__ == "__main__":
    cleanup_database()
