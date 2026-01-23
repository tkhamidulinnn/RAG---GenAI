from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from rag.embeddings import GeminiEmbeddings


def build_documents(pages: Iterable[dict]) -> List[Document]:
    documents: List[Document] = []
    for row in pages:
        text = row.get("text", "")
        if not text.strip():
            continue
        metadata = {"page": row.get("page"), "source": "pdf_text"}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def build_faiss(
    documents: List[Document],
    embeddings: GeminiEmbeddings,
    out_dir: Path,
) -> FAISS:
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(str(out_dir))
    return db


def build_qdrant(
    documents: List[Document],
    embeddings: GeminiEmbeddings,
    path: Path,
    collection: str,
) -> QdrantVectorStore:
    path.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            path=str(path),
            collection_name=collection,
        )
    except RuntimeError as e:
        if "already accessed" in str(e):
            raise RuntimeError(
                f"Qdrant storage is locked: {path}\n"
                "Stop Streamlit (Ctrl+C) before running ingest.py"
            ) from e
        raise
