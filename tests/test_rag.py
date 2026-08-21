from src.ingestion import extract_pdf_pages
from src.hybrid_retrieval import retrieve_hybrid
from src.chunking import chunk_pages
from src.llm import generate_answer, build_context
from src.vector_store import VectorStore
from src.embeddings import create_embeddings

PDF_PATH = "data/Ozempic.pdf"

# 1 Exptract pages from pdf
pages = extract_pdf_pages(PDF_PATH)

print("pages extracted from pdf: ", len(pages))

# 2 chunk pages into smaller pieces
chunks = chunk_pages(pages, chunk_size=1000)
l=0
if chunks:
    l = len(chunks)
print("chunks created: ", l)

# 3 create embeddings for chunks
texts = [chunk["text"] for chunk in chunks]

embed = create_embeddings(texts)

print("embeddings shape: ", embed.shape)

#4 build vector store
vector_store = VectorStore()
vector_store.build(embed, chunks)

#5 question

question = input("\nEnter your question: ")

# 6 retrieve 

res = retrieve_hybrid(question, vector_store, top_k=5)

print("\n retrieved sources: ", len(res))

print("\nRETRIEVED CHUNKS")
print("================")

for i, result in enumerate(res, start=1):

    print(f"\n--- RESULT {i} ---")
    print(f"Document : {result['document']}")
    print(f"Page     : {result['page']}")
    print(f"Score    : {result['score']:.3f}")

    print("\nText:")
    print(result["text"][:700])

# 7 build context

context = build_context(res)

#8 generate answer


if not res:
    answer = "I could not find this information in the provided drug documentation."
else:
    answer = generate_answer(question, context)

print("\nAnswer: ", answer)

print("\nSources")
for result in res:
    print(f"""{result['document']} 
        Page: {result['page']}""")
