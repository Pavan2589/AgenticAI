from pathlib import Path


def load_txt(file_path: str) -> list[dict]:
    path = Path(file_path)

    text = path.read_text(encoding="utf-8")

    return [{
        "document_id": path.stem,
        "source": path.name,
        "page_number": None,
        "text": text
    }]