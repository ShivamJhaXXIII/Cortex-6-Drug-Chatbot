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

vector_store = VectorStore()

vector_store.build(
    embeddings,
    chunks
)


# -------------------------
# Questions
# -------------------------

questions = [
    "What is the recommended starting dosage of Ozempic?",
    "What are the contraindications of Ozempic?",
    "What are the most common adverse reactions?",
    "What warnings and precautions are associated with Ozempic?",
    "What are the drug interactions?",
]


# -------------------------
# Test
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
        top_k=5,
        candidate_k=15
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n-------------------------")
        print(f"RESULT {i}")

        print(
            "Semantic :",
            round(
                result["semantic_score"],
                3
            )
        )

        print(
            "Keyword  :",
            round(
                result["keyword_score"],
                3
            )
        )

        print(
            "Combined :",
            round(
                result["combined_score"],
                3
            )
        )

        print(
            "Page     :",
            result["page"]
        )

        print(
            "Section  :",
            result.get(
                "section",
                "Unknown"
            )
        )

        print(
            "Text     :",
            result["text"][:500]
            .replace("\n", " ")
        )