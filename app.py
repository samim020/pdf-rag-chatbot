import os

import streamlit as st
from dotenv import load_dotenv

from src.ingestion import chunk_documents, load_uploaded_pdfs
from src.rag_chain import answer_question
from src.vectorstore import build_vectorstore, get_embedding_model

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")


def init_session_state():
    defaults = {
        "vectorstore": None,
        "indexed_files": [],
        "chunk_count": 0,
        "messages": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

# --- Sidebar: Upload & Configuration ---
with st.sidebar:
    st.header("Documents & Settings")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    with st.expander("Pipeline Settings", expanded=False):
        model_name = st.selectbox(
            "Groq Model",
            options=["openai/gpt-oss-20b"],
            index=0,
        )
        chunk_size = st.slider("Chunk Size", 300, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)
        top_k = st.slider("Retrieved Chunks (Top-K)", 1, 10, 4)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.05)

    if st.button("Process & Index PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Extracting text, chunking, and building FAISS index..."):
                docs, empty_files = load_uploaded_pdfs(uploaded_files)

                if empty_files:
                    st.warning(
                        f"No extractable text found in: {', '.join(empty_files)} "
                        "(possibly scanned/image-only PDFs)."
                    )

                if not docs:
                    st.error("None of the uploaded PDFs contained readable text.")
                else:
                    chunks = chunk_documents(
                        docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    embeddings = load_embeddings()
                    st.session_state.vectorstore = build_vectorstore(chunks, embeddings)
                    st.session_state.indexed_files = [
                        f.name for f in uploaded_files if f.name not in empty_files
                    ]
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.messages = []
                    st.success(
                        f"Indexed {len(st.session_state.indexed_files)} PDF(s) "
                        f"into {len(chunks)} chunks."
                    )

    if st.session_state.indexed_files:
        st.divider()
        st.subheader("Active Knowledge Base")
        st.caption(f"Total chunks: **{st.session_state.chunk_count}**")
        for fname in st.session_state.indexed_files:
            st.markdown(f"- `{fname}`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Reset Index", use_container_width=True):
                st.session_state.vectorstore = None
                st.session_state.indexed_files = []
                st.session_state.chunk_count = 0
                st.session_state.messages = []
                st.rerun()

# --- Main Chat Area ---
st.title("PDF RAG Assistant")

if not GROQ_API_KEY:
    st.error("`GROQ_API_KEY` not found. Add `GROQ_API_KEY=your_key` to your `.env` file and restart the app.")
    st.stop()

if st.session_state.vectorstore is None:
    st.info("Upload one or more PDFs in the sidebar and click **Process & Index PDFs** to start asking questions.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Retrieved Sources"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**{i}. `{src['source']}` — Page {src['page']}** "
                            f"(Similarity: `{src['score']:.2f}`)"
                        )
                        st.caption(src["snippet"])
                        if i < len(msg["sources"]):
                            st.divider()

    if prompt := st.chat_input("Ask a question about your uploaded PDFs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    result = answer_question(
                        query=prompt,
                        vectorstore=st.session_state.vectorstore,
                        groq_api_key=GROQ_API_KEY,
                        model_name=model_name,
                        temperature=temperature,
                        top_k=top_k,
                        chat_history=st.session_state.messages[:-1],
                    )
                    st.markdown(result["answer"])
                    with st.expander("Retrieved Sources"):
                        for i, src in enumerate(result["sources"], start=1):
                            st.markdown(
                                f"**{i}. `{src['source']}` — Page {src['page']}** "
                                f"(Similarity: `{src['score']:.2f}`)"
                            )
                            st.caption(src["snippet"])
                            if i < len(result["sources"]):
                                st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )
                except Exception as e:
                    st.error(f"Error generating answer: {e}")