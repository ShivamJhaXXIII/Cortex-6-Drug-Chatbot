def chunk_pages(pages, chunk_size=1000, overlap=150):
    """
    Chunk the text of each page into smaller chunks.
    Eachh chunk contains
    - Document name
    - Page number
    """
    chunks = []

    step = chunk_size - overlap

    for page in pages:

        text = page["text"]

        for start in range(0, len(text), step):

            chunk_text = text[start:start + chunk_size].strip()

            if not chunk_text:
                continue

            chunks.append({
                "text": chunk_text,
                "document": page["document"],
                "page": page["page"]
            })

    return chunks