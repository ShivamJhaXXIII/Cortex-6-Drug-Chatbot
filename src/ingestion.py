import fitz
import os
from pathlib import Path

def extract_pdf_pages(pdf_paths):
    """
    Extract text page by page from one or more PDFs.

    A single path is still accepted for backwards compatibility. When
    multiple paths are provided, pages from every PDF are returned in order.

    Returns : 
    [
        {
            "text": "...",
            "page": 1,
            "document": "drug.pdf"
        },
        ...
    ]
    """
    pages = []

    if isinstance(pdf_paths, (str, os.PathLike)):
        pdf_paths = [pdf_paths]

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        document = fitz.open(pdf_path)
        try:
            for page_number in range(document.page_count):
                page = document.load_page(page_number)
                text = str(page.get_text("text", sort=True))
                if text.strip():
                    pages.append(
                        {
                            "text": text,
                            "page": page_number + 1,
                            "document": pdf_path.name,
                        }
                    )
        finally:
            document.close()

    return pages