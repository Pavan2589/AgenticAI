"""
Document loading for .txt, .md and .pdf files.

PDF extraction uses PyMuPDF (fitz) - already installed in this repo's
venv and already the library Part 8 used
(part08_document_ingestion/loaders/pdf_loader.py) - instead of pypdf,
to avoid adding a redundant dependency. Page numbers are preserved
for PDFs; plain text/markdown files have no page concept, so
page_number is None for them.
"""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(file_path: str) -> list[dict]:
    """
    Returns a list of {"document_id", "source", "page_number", "text"}.
    One entry per PDF page, or a single entry for txt/md files.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}' for {path.name}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".pdf":
        return _load_pdf(path)

    return _load_text(path)


def _load_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")

    return [{
        "document_id": path.stem,
        "source": path.name,
        "page_number": None,
        "text": text,
    }]


def _load_pdf(path: Path) -> list[dict]:
    import fitz  # PyMuPDF

    pages = []

    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pages.append({
                "document_id": path.stem,
                "source": path.name,
                "page_number": page_number,
                "text": page.get_text("text"),
            })

    return pages
