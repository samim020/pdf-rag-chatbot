import os
import tempfile
from typing import List, Tuple

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_uploaded_pdfs(uploaded_files) -> Tuple[List[Document], List[str]]:
    """
    Loads Streamlit UploadedFile PDFs using PyMuPDFLoader.
    Preserves the original filename and 1-indexed page numbers in metadata.
    Returns (documents, empty_files).
    """
    all_documents: List[Document] = []
    empty_files: List[str] = []

    for uploaded_file in uploaded_files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()

            valid_pages = []
            for doc in pages:
                if doc.page_content and doc.page_content.strip():
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
                    valid_pages.append(doc)

            if not valid_pages:
                empty_files.append(uploaded_file.name)
            else:
                all_documents.extend(valid_pages)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return all_documents, empty_files


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Splits page-level Document objects into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)