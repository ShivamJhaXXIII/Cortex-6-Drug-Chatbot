from src.hybrid_retrieval import retrieve_hybrid
from src.llm import generate_answer, build_context, rewrite_query
from groq import Groq
import streamlit as st

class RAGEngine:

    def __init__(self, vector_store):

        self.vector_store = vector_store
        self.groq_client = Groq(api_key=st.secrets["API_KEY"])
        self.model = "openai/gpt-oss-120b"

    def ask(self, question, history=None):
        history = history or []
        if history:
            question = rewrite_query(question, history, self.groq_client, self.model)

        results = retrieve_hybrid(
            question,
            self.vector_store,
            top_k=5,
            candidate_k=15
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
            context,
            history
        )

        sources = []

        for result in results:

            sources.append({
                "document": result["document"],
                "page": result["page"],
                "section": result["section"],
                "text": result["text"]
            })

        return {
            "answer": answer,
            "sources": sources
        }