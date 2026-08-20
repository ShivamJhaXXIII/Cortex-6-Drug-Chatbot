import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, embeddings, chunks):
        embeddings = np.array(embeddings, dtype="float32")
        
        # Ensure 2D array shape (N, dimension)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query_embedding, top_k=5):
        # Guard clause: throw explicit error if index isn't built yet
        if self.index is None:
            raise RuntimeError("VectorStore index is not built yet. Call 'build()' first.")

        query_embedding = np.array(query_embedding, dtype="float32")
        
        # Ensure query is 2D shape (1, dimension) for FAISS
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            chunk = self.chunks[index].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results