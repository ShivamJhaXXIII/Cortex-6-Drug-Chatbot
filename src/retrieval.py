from src.embeddings import create_emebeddings

def retrieve(question, vector_store, top_k=5, min_score=0.30):
    """
    Retrieve relevant chunks for a question
    """
    query_embedding = create_emebeddings([question])[0]

    results = vector_store.search(query_embedding, top_k=top_k)

    results = [result for result in results if result["score"] >= min_score]

    return results