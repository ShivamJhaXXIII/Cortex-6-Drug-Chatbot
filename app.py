import os
import hashlib
import streamlit as st

from src.ingestion import extract_pdf_pages
from src.chunking import chunk_pages
from src.embeddings import create_embeddings
from src.vector_store import VectorStore
from src.rag import RAGEngine
from src.web_ingestion import fetch_official_source


st.set_page_config(
    page_title="Drug Information Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# CSS — minimal; we rely on st.container(height=) for scrollable regions
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Hide default top padding Streamlit adds */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }

    /* Remove the extra gap Streamlit puts above the first element */
    [data-testid="stAppViewBlockContainer"] > div:first-child {
        margin-top: 0;
    }

    /* Make column wrappers stretch so panels look even */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }

    /* Panel card styling */
    .panel-card {
        border: 1px solid #dbe3ea;
        border-radius: 12px;
        padding: 0.75rem 1rem 0.6rem;
        background: #ffffff;
        margin-bottom: 0.5rem;
    }

    /* Tighten chat message avatars */
    [data-testid="stChatMessage"] {
        padding: 0.4rem 0.2rem;
    }

    /* Remove border from st.container(height=) scrollable box */
    [data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants — tune these to match your screen / header size
# ---------------------------------------------------------------------------
PANEL_HEIGHT = 560                # px — chat messages scroll area
SOURCES_RETRIEVED_HEIGHT = 460    # px — retrieved chunks scroll area


# ---------------------------------------------------------------------------
# RAG engine builder
# ---------------------------------------------------------------------------

@st.cache_resource
def build_engine(pdf_paths):
    pages = extract_pdf_pages(pdf_paths)
    chunks = chunk_pages(pages)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)
    vector_store = VectorStore()
    vector_store.build(embeddings, chunks)
    return RAGEngine(vector_store)


def rebuild_engine_from_documents(uploaded_documents):
    document_signature = tuple(
        (document["name"], document["hash"])
        for document in uploaded_documents
    )

    if st.session_state.document_signature == document_signature:
        return False

    pdf_paths = []
    for index, document in enumerate(
        uploaded_documents,
        start=1,
    ):
        pdf_path = os.path.join(
            "storage",
            f"{document['hash'][:12]}_{index}_{document['name']}",
        )
        with open(pdf_path, "wb") as f:
            f.write(document["content"])
        pdf_paths.append(pdf_path)

    with st.spinner("Processing documents…"):
        st.session_state.engine = build_engine(tuple(pdf_paths))

    st.session_state.uploaded_documents = uploaded_documents
    st.session_state.document_names = [
        document["name"] for document in uploaded_documents
    ]
    st.session_state.document_signature = document_signature
    st.session_state.messages = []
    st.session_state.sources = []
    st.session_state.show_uploader = False
    st.success(
        f"Processed {len(uploaded_documents)} document(s) successfully."
    )
    st.rerun()

    return True


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("messages", []),
    ("sources", []),
    ("engine", None),
    ("document_names", []),
    ("uploaded_documents", []),
    ("document_signature", None),
    ("show_uploader", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("💊 Drug Information Assistant")
st.caption("Grounded answers from drug documentation")

# ---------------------------------------------------------------------------
# Main layout  —  Sources (left)  |  Chat (right)
# ---------------------------------------------------------------------------

sources_col, chat_col = st.columns([1, 1.8], gap="large")


# ══════════════════════════════════════════════════════════════
# LEFT — SOURCES
# ══════════════════════════════════════════════════════════════

with sources_col:
    # Header row: title + "Add source" button side by side
    title_col, btn_col = st.columns([2, 1])
    with title_col:
        st.markdown("### Sources")
    with btn_col:
        st.markdown("<div style='padding-top:0.6rem'>", unsafe_allow_html=True)
        if st.button("＋ Add source", use_container_width=True):
            st.session_state.show_uploader = not st.session_state.show_uploader
        st.markdown("</div>", unsafe_allow_html=True)

    # Inline uploader — toggled by the button
    if st.session_state.show_uploader:
        uploaded_files = st.file_uploader(
            "Upload drug documentation (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        st.subheader("Add source")
        source_type = st.radio(
            "Source type",
            ["Upload PDF", "Official URL"],
            horizontal=True,
            key="source_type_selector",
        )

        if source_type == "Official URL":
            url = st.text_input(
                "Official drug documentation URL",
                placeholder="https://dailymed.nlm.nih.gov/..."
            )

            add_url = st.button(
                "Add source",
                key="add_official_url_source"
            )

            if add_url and url:
                try:
                    with st.spinner("Fetching official document..."):
                        pdf_path = fetch_official_source(url)

                    with open(pdf_path, "rb") as f:
                        content = f.read()

                    file_hash = hashlib.sha256(content).hexdigest()
                    uploaded_documents = list(st.session_state.uploaded_documents)
                    known_hashes = {document["hash"] for document in uploaded_documents}

                    if file_hash in known_hashes:
                        st.info("This source is already added.")
                    else:
                        uploaded_documents.append({
                            "name": os.path.basename(pdf_path),
                            "content": content,
                            "hash": file_hash,
                        })
                        rebuild_engine_from_documents(uploaded_documents)
                except Exception as exc:
                    st.error(f"Could not add source: {exc}")

        if uploaded_files:
            os.makedirs("storage", exist_ok=True)
            uploaded_documents = list(st.session_state.uploaded_documents)
            known_hashes = {document["hash"] for document in uploaded_documents}

            for uploaded_file in uploaded_files:
                content = uploaded_file.getvalue()
                file_hash = hashlib.sha256(content).hexdigest()
                if file_hash not in known_hashes:
                    uploaded_documents.append({
                        "name": os.path.basename(uploaded_file.name),
                        "content": content,
                        "hash": file_hash,
                    })
                    known_hashes.add(file_hash)

            rebuild_engine_from_documents(uploaded_documents)

    # Uploaded document list
    if st.session_state.document_names:
        for document_name in st.session_state.document_names:
            st.markdown(f"📄 `{document_name}`")
    else:
        st.caption("No document added yet.")

    st.markdown("**Retrieved evidence**")

    # Scrollable container for retrieved chunks
    with st.container(height=SOURCES_RETRIEVED_HEIGHT, border=True):
        if not st.session_state.sources:
            st.info("Evidence chunks will appear here after each response.")
        else:
            for i, source in enumerate(st.session_state.sources, start=1):
                with st.container(border=True):
                    st.markdown(
                        f"**Source {i}** · `{source['document']}` · Page {source['page']}"
                    )
                    if source.get("section"):
                        st.caption(source["section"])
                    with st.expander("View text"):
                        st.write(source["text"])


# ══════════════════════════════════════════════════════════════
# RIGHT — CHAT
# ══════════════════════════════════════════════════════════════

with chat_col:
    st.markdown("### Chat")

    # Scrollable message history — fixed height, never grows
    with st.container(height=PANEL_HEIGHT, border=True):
        if st.session_state.engine is None:
            st.info("Click **＋ Add source** in the Sources panel to upload documents.")
        else:
            if not st.session_state.messages:
                st.caption("Ask a question below to get started.")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Input bar — always below the scrollable box
    with st.form(key="chat_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question",
            placeholder="Ask a question about the uploaded documents…",
            label_visibility="collapsed",
            disabled=(st.session_state.engine is None),
        )
        submitted = st.form_submit_button(
            "Send",
            use_container_width=True,
            disabled=(st.session_state.engine is None),
        )


# ---------------------------------------------------------------------------
# Handle submission
# ---------------------------------------------------------------------------

if submitted and question and st.session_state.engine is not None:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Searching the document…"):
        result = st.session_state.engine.ask(
            question,
            st.session_state.messages[:-1]
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
    })

    st.session_state.sources = result["sources"]
    st.rerun()