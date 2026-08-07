"""Build (or rebuild) the Chroma vector store from the knowledge_base folder.

Run manually:  python -m app.rag.ingest
"""
from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.config import CHROMA_DIR, EMBEDDING_MODEL, KNOWLEDGE_DIR

COLLECTION = "everline_kb"

_HEADERS = [("#", "topic"), ("##", "section")]


def load_documents() -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    docs: list[Document] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            chunk.metadata["source"] = path.name
            docs.append(chunk)
    return docs


def build() -> int:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    docs = load_documents()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )
    return len(docs)


if __name__ == "__main__":
    count = build()
    print(f"Ingested {count} chunks into Chroma at {CHROMA_DIR}")
