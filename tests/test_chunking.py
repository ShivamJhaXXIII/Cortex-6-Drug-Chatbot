from src.ingestion import extract_pdf_pages
from src.chunking import chunk_pages


PDF_PATH = "data/Ozempic.pdf"


pages = extract_pdf_pages(PDF_PATH)

print("Pages:", len(pages))


chunks = chunk_pages(pages)

print("Chunks:", len(chunks))


for i, chunk in enumerate(chunks):

    if chunk["section"] != "Unknown":

        print(
            i + 1,
            "| Page:",
            chunk["page"],
            "| Section:",
            chunk["section"]
        )