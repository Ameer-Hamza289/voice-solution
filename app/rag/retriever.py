"""Query interface over the persisted Chroma knowledge base."""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import CHROMA_DIR, EMBEDDING_MODEL
from app.rag.ingest import COLLECTION


@lru_cache(maxsize=1)
def _store() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def search(query: str, k: int = 4) -> str:
    """Return the top-k knowledge chunks formatted for the LLM."""
    results = _store().similarity_search(query, k=k)
    if not results:
        return "No relevant company information found."
    parts = []
    for doc in results:
        source = doc.metadata.get("source", "kb")
        parts.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
