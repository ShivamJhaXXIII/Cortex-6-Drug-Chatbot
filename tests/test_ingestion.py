from src.chunking import chunk_pages
from src.ingestion import extract_pdf_pages
import fitz


PDF_PATH = "data/Ozempic.pdf"

doc = fitz.open(PDF_PATH)

print("TOTAL PDF PAGES:", len(doc))
print("\nPAGE TEXT LENGTHS:")

for i, page in enumerate(doc, start=1):
    text = page.get_text("text")

    print(
        f"Page {i:02d}: "
        f"{len(text.strip()):5d} characters"
    )

doc.close()


pages = extract_pdf_pages(PDF_PATH)
print("Extracted pages:", len(pages))

for page in pages[:10]:
    print(
        page["page"],
        len(page["text"])
    )

chunks = chunk_pages(pages)
print("total chunks: ",len(chunks))

print("\nEXTRACTED PAGES:", len(pages))