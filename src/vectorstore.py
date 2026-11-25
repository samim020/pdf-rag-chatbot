from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    """Initializes a local Sentence-Transformers embedding model with unit normalization."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Creates an in-memory FAISS index from document chunks."""
    return FAISS.from_documents(chunks, embeddings)


def retrieve_chunks(
    vectorstore: FAISS,
    query: str,
    top_k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Retrieves top_k chunks along with cosine similarity scores (0.0 to 1.0).
    With unit-normalized embeddings, FAISS IndexFlatL2 returns squared L2 distance d^2,
    where cosine_similarity = 1 - (d^2 / 2).
    """
    raw_results = vectorstore.similarity_search_with_score(query, k=top_k)
    scored_results: List[Tuple[Document, float]] = []

    for doc, l2_sq_dist in raw_results:
        cosine_sim = max(0.0, min(1.0, 1.0 - (float(l2_sq_dist) / 2.0)))
        scored_results.append((doc, cosine_sim))

    return scored_results