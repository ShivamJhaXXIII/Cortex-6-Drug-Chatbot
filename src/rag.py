from src.hybrid_retrieval import retrieve_hybrid
from src.llm import generate_answer, build_context

class RAGEngine:

    def __init__(self, vector_store, chunks):

        self.vector_store = vector_store
        self.chunks = chunks

    def ask(self, question):

        results = retrieve_hybrid(
            question,
            self.vector_store,
            self.chunks,
            top_k=3,
            candidate_k=10
        )

        if not results:

            return {
                "answer": (
                    "I could not find this information "
                    "in the provided drug documentation."
                ),
                "sources": []
            }

        context = build_context(results)

        answer = generate_answer(
            question,
            context
        )

        sources = []

        for result in results:

            sources.append({
                "document": result["document"],
                "page": result["page"],
                "section": result.get(
                    "section",
                    "Unknown"
                ),
                "text": result["text"]
            })

        return {
            "answer": answer,
            "sources": sources
        }