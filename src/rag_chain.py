from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.vectorstore import retrieve_chunks

SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Answer the user's question using ONLY the provided context excerpts from the uploaded PDFs.
- Cite sources inline using the format (filename, p. X) whenever referencing specific facts.
- If the context does not contain enough information to answer the question, state clearly that the uploaded documents do not cover it.
- Keep answers clear, direct, and well-structured."""


def _format_context(scored_docs) -> str:
    blocks = []
    for idx, (doc, score) in enumerate(scored_docs, start=1):
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page_number", "?")
        blocks.append(
            f"[Excerpt {idx} | Source: {source} | Page: {page} | Relevance: {score:.2f}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(
    query: str,
    vectorstore: FAISS,
    groq_api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    top_k: int = 4,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Runs retrieval + grounded generation and returns the answer and source metadata."""
    scored_docs = retrieve_chunks(vectorstore, query, top_k=top_k)
    context_text = _format_context(scored_docs)

    llm = ChatGroq(
        api_key=groq_api_key,
        model=model_name,
        temperature=temperature,
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Include the last 3 Q&A turns for conversational continuity
    if chat_history:
        for turn in chat_history[-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    user_payload = (
        f"Context from uploaded PDFs:\n\n{context_text}\n\n"
        f"Question: {query}"
    )
    messages.append(HumanMessage(content=user_payload))

    response = llm.invoke(messages)

    sources = [
        {
            "source": doc.metadata.get("source", "Unknown PDF"),
            "page": doc.metadata.get("page_number", "?"),
            "score": round(score, 3),
            "snippet": doc.page_content.strip(),
        }
        for doc, score in scored_docs
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }