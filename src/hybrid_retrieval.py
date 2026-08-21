import re
from src.query_router import (
    detect_intent,
    expected_section
)
from src.embeddings import create_embeddings


STOP_WORDS = {
    "what",
    "are",
    "is",
    "the",
    "of",
    "and",
    "or",
    "for",
    "to",
    "a",
    "an",
    "with",
    "on",
    "in",
    "about",
    "associated",
    "does",
    "do",
    "how",
    "can",
}


def is_multi_source_question(question):
    """Return whether a question asks to compare multiple sources."""
    question = question.lower()

    comparison_terms = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "versus",
        "vs",
        "both",
        "each",
        "between",
        "respectively",
    ]

    return any(term in question for term in comparison_terms)


def tokenize(text):
    """
    Convert text into a simple set of lowercase alphanumeric tokens.
    """
    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


def keyword_score(question_tokens, chunk):
    """
    Calculate lexical overlap (intersection) between pre-tokenized question words
    and the chunk text, normalized by the number of query tokens.
    
    Args:
        question_tokens (set): Pre-tokenized set of non-stop-word query terms.
        chunk (dict): The dictionary representation of the document chunk.
        
    Returns:
        float: Normalized lexical overlap score between 0.0 and 1.0.
    """
    if not question_tokens:
        return 0.0

    # Tokenize the target chunk text
    chunk_tokens = tokenize(chunk["text"])

    # Intersect query tokens with chunk tokens to find common terms
    overlap = question_tokens & chunk_tokens

    # Lexical score is the percentage of query tokens present in this chunk
    return len(overlap) / len(question_tokens)


def retrieve_hybrid(
    question,
    vector_store,
    top_k=5,
    candidate_k=15,
    semantic_weight=0.65,
    keyword_weight=0.25,
    section_weight=0.10
):
    """
    Retrieve the most relevant document chunks for a question using a hybrid approach
    that combines semantic similarity, lexical overlap, and section-based routing.
    
    Args:
        question (str): The search query.
        vector_store (VectorStore): The vector database interface.
        top_k (int): Number of final results to return.
        candidate_k (int): Number of candidate chunks to fetch initially via semantic search.
        semantic_weight (float): Weighted importance of the dense embedding score.
        keyword_weight (float): Weighted importance of the lexical matching score.
        section_weight (float): Weighted importance of routing to the predicted section.
        
    Returns:
        list: Up to top_k chunks sorted by their combined hybrid score.
    """
    # Detect the query intent (e.g., dosage, side effects) to route it to the expected section
    intent = detect_intent(question)
    target_section = expected_section(intent)

    # -----------------------------------------------------------------
    # Optimization: Tokenize and filter query stop words once outside the loop.
    # This prevents redundant parsing for each candidate chunk.
    # -----------------------------------------------------------------
    question_tokens = tokenize(question) - STOP_WORDS

    # -------------------------
    # 1. Semantic retrieval
    # -------------------------
    # Embed the query string
    query_embedding = create_embeddings([question])[0]

    # Comparison questions need a wider pool so multiple documents can contribute.
    if is_multi_source_question(question):
        candidate_k = max(candidate_k, top_k * 3)

    # Fetch top semantic candidates using cosine similarity (via FlatIP)
    semantic_results = vector_store.search(
        query_embedding,
        candidate_k
    )

    # -------------------------
    # 2. Calculate combined score
    # -------------------------
    scored_results = []

    for result in semantic_results:
        # Filter out non-retrievable chunks (e.g., packaging pages)
        if not result.get("retrievable", True):
            continue

        # Cosine similarity score
        semantic = result["score"]

        # Lexical score using the pre-tokenized query tokens
        lexical = keyword_score(
            question_tokens,
            result
        )

        # Section bonus: 1.0 if the chunk's section matches the detected query intent section, else 0.0
        section_bonus = 0.0
        if (
            target_section
            and result.get("section") == target_section
        ):
            section_bonus = 1.0

        # Calculate the final combined hybrid score based on weights
        combined = (
            semantic_weight * semantic
            + keyword_weight * lexical
            + section_weight * section_bonus
        )

        # Create a new dictionary result with individual scores included
        result_copy = result.copy()
        result_copy["section"] = result.get("section", "Unknown")
        result_copy["semantic_score"] = semantic
        result_copy["keyword_score"] = lexical
        result_copy["combined_score"] = combined

        scored_results.append(result_copy)

    # -------------------------
    # 3. Rank
    # -------------------------
    # Sort candidates in descending order based on the combined score
    scored_results.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    # Return top K sorted results
    return select_balanced_results(
        scored_results,
        top_k=top_k
    )

def select_balanced_results(
    results,
    top_k=6
):
    """
    Select evidence while giving each document
    a chance to contribute.
    """

    selected = []

    documents = []

    for result in results:

        document = result["document"]

        if document not in documents:
            documents.append(document)

    # First: take the best result from each document
    for document in documents:

        for result in results:

            if (
                result["document"] == document
            ):

                selected.append(result)
                break

    # Second: fill remaining slots by score
    for result in results:

        if len(selected) >= top_k:
            break

        if result not in selected:
            selected.append(result)

    # Final ordering by score
    selected.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return selected[:top_k]