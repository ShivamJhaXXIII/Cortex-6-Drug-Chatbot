from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings

def embedding_stats(embeddings):

    import numpy as np

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    print(
        "Min norm:",
        norms.min()
    )

    print(
        "Max norm:",
        norms.max()
    )

    print(
        "Mean norm:",
        norms.mean()
    )