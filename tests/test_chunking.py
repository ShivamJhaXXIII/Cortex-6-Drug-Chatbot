from src.ingestion import extract_pdf_pages
from src.chunking import chunk_pages


PDF_PATH = "data/Ozempic.pdf"


pages = extract_pdf_pages(PDF_PATH)

print("Pages:", len(pages))


chunks = chunk_pages(pages)

print("Chunks:", len(chunks))


print("\nFirst 5 chunks:")

for i, chunk in enumerate(chunks[:5]):

    print("\n--------------------")

    print("Chunk:", i + 1)
    print("Document:", chunk["document"])
    print("Page:", chunk["page"])
    print("Characters:", len(chunk["text"]))

    print(chunk["text"][:300])