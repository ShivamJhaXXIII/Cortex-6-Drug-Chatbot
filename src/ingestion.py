import fitz
from pathlib import Path

def extract_pdf_pages(pdf_path: str):
    """
    Extract text page by page from a PDF
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
    pdf_path = Path(pdf_path)
    document = fitz.open(pdf_path)
    pages = []


    for page_number in range(document.page_count):
        page = document.load_page(page_number)
        text = str(page.get_text())
        if text.strip(): # Only include pages with text
            pages.append(
                {
                    "text": text,
                    "page": page_number + 1,
                    "document": pdf_path.name,
                }
            )
    document.close()
    return pages