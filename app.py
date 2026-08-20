import os
import streamlit as st

from src.ingestion import extract_pdf_pages
from src.chunking import chunk_pages
from src.embeddings import create_embeddings
from src.vector_store import VectorStore
from src.rag import RAGEngine


st.set_page_config(
    page_title="Drug Information Assistant",
    page_icon="💊",
    layout="wide"
)


# -------------------------
# Title
# -------------------------

st.title("Drug Information Assistant")

st.caption(
    "Grounded answers from drug documentation"
)


# -------------------------
# Build RAG engine
# -------------------------

@st.cache_resource
def build_engine(pdf_path):

    pages = extract_pdf_pages(pdf_path)

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

    return RAGEngine(
        vector_store,
        chunks
    )


# -------------------------
# Session state
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "engine" not in st.session_state:
    st.session_state.engine = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# -------------------------
# Upload
# -------------------------

uploaded_file = st.file_uploader(
    "Upload drug documentation",
    type=["pdf"]
)


if uploaded_file:

    os.makedirs(
        "storage",
        exist_ok=True
    )

    pdf_path = os.path.join(
        "storage",
        uploaded_file.name
    )

    # Avoid processing same document repeatedly
    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with open(pdf_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        with st.spinner(
            "Processing document..."
        ):

            st.session_state.engine = (
                build_engine(pdf_path)
            )

        st.session_state.document_name = (
            uploaded_file.name
        )

        st.session_state.messages = []
        st.session_state.sources = []

        st.success(
            "Document processed successfully."
        )


# -------------------------
# Main layout
# -------------------------

left_col, right_col = st.columns(
    [1, 2.3],
    gap="large"
)


# ==================================================
# LEFT — SOURCES
# ==================================================

with left_col:

    st.subheader("Sources")

    if st.session_state.document_name:

        st.caption(
            st.session_state.document_name
        )

    if not st.session_state.sources:

        st.info(
            "Sources used for the latest answer "
            "will appear here."
        )

    else:

        for i, source in enumerate(
            st.session_state.sources,
            start=1
        ):

            st.markdown(
                f"### Source {i}"
            )

            st.write(
                f"**Page:** {source['page']}"
            )

            section = source.get(
                "section",
                "Unknown"
            )

            if section != "Unknown":

                st.write(
                    f"**Section:** {section}"
                )

            # Evidence text
            if source.get("text"):

                with st.expander(
                    "View evidence",
                    expanded=i == 1
                ):

                    st.write(
                        source["text"]
                    )

            st.divider()


# ==================================================
# RIGHT — CHAT
# ==================================================

with right_col:

    st.subheader("Chat")

    if (
        st.session_state.engine
        is None
    ):

        st.info(
            "Upload a drug document to begin."
        )

    else:

        # Previous messages
        for message in (
            st.session_state.messages
        ):

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )


        question = st.chat_input(
            "Ask about the uploaded document..."
        )


        if question:

            # Store user message
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):

                st.markdown(question)


            # Generate response
            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Searching documentation..."
                ):

                    result = (
                        st.session_state
                        .engine
                        .ask(question)
                    )

                answer = result["answer"]

                st.markdown(answer)


            # Store assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


            # Update LEFT panel
            st.session_state.sources = (
                result["sources"]
            )

            st.rerun()