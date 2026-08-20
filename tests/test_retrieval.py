from src.ingestion import extract_pdf_pages
from src.chunking import chunk_pages
from src.embeddings import create_embeddings
from src.vector_store import VectorStore
from src.hybrid_retrieval import retrieve_hybrid


PDF_PATH = "data/Ozempic.pdf"


# -------------------------
# Build knowledge base
# -------------------------

pages = extract_pdf_pages(PDF_PATH)

chunks = chunk_pages(pages)

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = create_embeddings(texts)

from src.embeddings import embedding_stats

embedding_stats(embeddings)

vector_store = VectorStore()

vector_store.build(
    embeddings,
    chunks
)

import numpy as np

test_vector = embeddings[0].astype("float32")

print("\nVECTOR CHECK")
print("Embedding norm:", np.linalg.norm(test_vector))

stored_vector = vector_store.index.reconstruct(0)

print(
    "Stored norm:",
    np.linalg.norm(stored_vector)
)

print(
    "Max difference:",
    np.max(
        np.abs(test_vector - stored_vector)
    )
)

print(
    "Dot product:",
    np.dot(test_vector, stored_vector)
)

scores, indices = vector_store.index.search(
    test_vector.reshape(1, -1),
    5
)

print("\nRAW FAISS TEST")
print("Scores:", scores)
print("Indices:", indices)


# -------------------------
# Test questions
# -------------------------

questions = [
    "What is the recommended starting dosage of Ozempic?",
    "What are the contraindications of Ozempic?",
    "What are the most common adverse reactions?",
    "What warnings and precautions are associated with Ozempic?",
    "What are the drug interactions?",
]


# -------------------------
# Evaluate retrieval
# -------------------------

for question in questions:

    print("\n")
    print("=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    results = retrieve_hybrid(
        question,
        vector_store,
        chunks,
        top_k=5
    )

    for i, result in enumerate(results, start=1):

        print("\n-------------------------")
        print(f"RESULT {i}")

        print(
            "Score   :",
            round(result["score"], 3)
        )

        print(
            "Page    :",
            result["page"]
        )

        print(
            "Section :",
            result.get("section", "Unknown")
        )

        print(
            "Text    :",
            result["text"][:500].replace("\n", " ")
        )