import os
import hashlib
import streamlit as st

from src.ingestion import extract_pdf_pages
from src.chunking import chunk_pages
from src.embeddings import create_embeddings
from src.vector_store import VectorStore
from src.rag import RAGEngine


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
    return RAGEngine(vector_store, chunks)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("messages", []),
    ("sources", []),
    ("engine", None),
    ("document_names", []),
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
        if uploaded_files:
            os.makedirs("storage", exist_ok=True)
            uploaded_names = [os.path.basename(uploaded_file.name) for uploaded_file in uploaded_files]
            uploaded_contents = [uploaded_file.getvalue() for uploaded_file in uploaded_files]
            document_signature = tuple(
                (name, hashlib.sha256(content).hexdigest())
                for name, content in zip(uploaded_names, uploaded_contents)
            )

            if st.session_state.document_signature != document_signature:
                pdf_paths = []
                for index, (uploaded_file, name, content) in enumerate(
                    zip(uploaded_files, uploaded_names, uploaded_contents),
                    start=1,
                ):
                    file_hash = document_signature[index - 1][1][:12]
                    pdf_path = os.path.join("storage", f"{file_hash}_{index}_{name}")
                    with open(pdf_path, "wb") as f:
                        f.write(content)
                    pdf_paths.append(pdf_path)

                with st.spinner("Processing documents…"):
                    st.session_state.engine = build_engine(tuple(pdf_paths))

                st.session_state.document_names = uploaded_names
                st.session_state.document_signature = document_signature
                st.session_state.messages = []
                st.session_state.sources = []
                st.session_state.show_uploader = False
                st.success(f"Processed {len(uploaded_files)} document(s) successfully.")
                st.rerun()

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
        result = st.session_state.engine.ask(question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
    })

    st.session_state.sources = result["sources"]
    st.rerun()