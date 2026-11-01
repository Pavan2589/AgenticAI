import fitz
from pathlib import Path


def load_pdf(file_path: str) -> list[dict]:
    documents = []

    path = Path(file_path)

    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            documents.append({
                "document_id": path.stem,
                "source": path.name,
                "page_number": page_number,
                "text": text
            })

    return documents