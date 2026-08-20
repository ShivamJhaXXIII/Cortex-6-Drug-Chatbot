import streamlit as st
from src.ingestion import extract_pdf_pages
from src.chunking import chunk_pages
from src.embeddings import create_embeddings
from src.vector_store import VectorStore
from src.rag import RAGEngine


st.set_page_config(
    page_title="Drug Information Assistant",
    page_icon="💊"
)


st.title("Drug Information Assistant")

st.caption(
    "Ask questions about the uploaded drug documentation."
)


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
# PDF upload
# -------------------------

uploaded_file = st.file_uploader(
    "Upload drug documentation",
    type=["pdf"]
)


if uploaded_file:

    import os

    os.makedirs(
        "storage",
        exist_ok=True
    )

    pdf_path = os.path.join(
        "storage",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    with st.spinner(
        "Processing document..."
    ):

        engine = build_engine(
            pdf_path
        )

    st.success(
        "Document processed successfully."
    )


    # -------------------------
    # Question
    # -------------------------

    question = st.chat_input(
        "Ask a question about the document..."
    )


    if question:

        with st.chat_message("user"):

            st.write(question)


        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the document..."
            ):

                result = engine.ask(
                    question
                )

            st.write(
                result["answer"]
            )

            if result["sources"]:

                st.markdown(
                    "### Sources"
                )

                for source in result["sources"]:

                    st.write(
                        f"• {source['document']} "
                        f"— Page {source['page']}"
                    )