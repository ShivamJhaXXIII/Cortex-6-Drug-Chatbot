from src.ingestion import extract_pdf_pages


pdf_path = "data/Ozempic.pdf"

pages = extract_pdf_pages(pdf_path)

print("Number of pages extracted:", len(pages))

for page in pages[:2]:
    print("\n--------------------")
    print("Document:", page["document"])
    print("Page:", page["page"])
    print(page["text"][:1000])