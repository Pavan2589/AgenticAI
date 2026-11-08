import json
from pathlib import Path

from loaders.pdf_loader import load_pdf
from loaders.text_loader import load_txt
from loaders.cleaner import clean_text

from chunking.fixed_size import fixed_size_chunk
from chunking.recursive import recursive_chunk
from chunking.semantic import semantic_chunk


BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_documents() -> list[dict]:
    documents = []

    for file_path in DOCUMENTS_DIR.iterdir():

        if file_path.suffix.lower() == ".pdf":
            documents.extend(load_pdf(str(file_path)))

        elif file_path.suffix.lower() == ".txt":
            documents.extend(load_txt(str(file_path)))

    return documents


def create_chunks(documents: list[dict], strategy: str) -> list[dict]:

    all_chunks = []

    for document in documents:

        text = clean_text(document["text"])

        if not text:
            continue

        if strategy == "fixed":
            chunks = fixed_size_chunk(text)

        elif strategy == "recursive":
            chunks = recursive_chunk(text)

        elif strategy == "semantic":
            chunks = semantic_chunk(text)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for index, chunk in enumerate(chunks):

            all_chunks.append({
                "document_id": document["document_id"],
                "chunk_id": f"{document['document_id']}_{index}",
                "chunk_index": index,
                "source": document["source"],
                "page_number": document["page_number"],
                "strategy": strategy,
                "text": chunk
            })

    return all_chunks


def save_chunks(chunks: list[dict], strategy: str):

    output_file = OUTPUT_DIR / f"{strategy}_chunks.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved {len(chunks)} chunks to {output_file}")


def main():

    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} document pages/records")

    for strategy in ["fixed", "recursive", "semantic"]:

        print(f"\nRunning {strategy} chunking...")

        chunks = create_chunks(documents, strategy)

        save_chunks(chunks, strategy)


if __name__ == "__main__":
    main()