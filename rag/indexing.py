from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from rag.embeddings import GeminiEmbeddings


def build_documents(
    pages: Iterable[dict],
    extract_metadata: bool = False,
) -> List[Document]:
    """Build LangChain Document objects from page data.

    Args:
        pages: Iterable of dicts with "page" and "text" keys
        extract_metadata: If True, extract section and content type metadata
                         using the metadata_extractor module (Practice 3)

    Returns:
        List of Document objects
    """
    documents: List[Document] = []

    # Convert to list for potential double-pass (metadata extraction)
    pages_list = list(pages)

    # Build page metadata map if extraction is enabled
    page_metadata_map = {}
    if extract_metadata:
        from rag.metadata_extractor import build_page_metadata_map
        page_metadata_map = build_page_metadata_map(pages_list)

    for row in pages_list:
        text = row.get("text", "")
        if not text.strip():
            continue

        page_num = row.get("page", 1)
        metadata = {"page": page_num, "source": "pdf_text"}

        # Add extracted metadata if available
        if page_num in page_metadata_map:
            page_meta = page_metadata_map[page_num]
            metadata["section"] = page_meta.section
            metadata["content_type"] = page_meta.content_type
            metadata["has_tables"] = page_meta.has_tables
            metadata["has_figures"] = page_meta.has_figures

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    enrich_metadata: bool = False,
) -> List[Document]:
    """Split documents into chunks.

    Args:
        documents: List of Document objects
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        enrich_metadata: If True, enrich chunk metadata with content type detection

    Returns:
        List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    if enrich_metadata:
        from rag.metadata_extractor import extract_chunk_metadata

        # Track chunk index per page
        chunk_counts: Dict[int, int] = {}

        for chunk in chunks:
            page = chunk.metadata.get("page", 1)
            section = chunk.metadata.get("section", "Unknown")

            chunk_index = chunk_counts.get(page, 0)
            chunk_counts[page] = chunk_index + 1

            chunk_meta = extract_chunk_metadata(
                chunk.page_content,
                page,
                section,
                chunk_index,
            )

            # Update with chunk-specific metadata
            chunk.metadata["content_type"] = chunk_meta.content_type
            chunk.metadata["chunk_index"] = chunk_meta.chunk_index
            chunk.metadata["has_numbers"] = chunk_meta.has_numbers
            chunk.metadata["has_dates"] = chunk_meta.has_dates

    return chunks


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
    url: str | None = None,
) -> QdrantVectorStore:
    """Build Qdrant index. Use url= for Docker (Qdrant server); else use path (local storage)."""
    if url:
        return QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            url=url,
            collection_name=collection,
        )
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


def build_bm25_index(
    documents: List[Document],
    save_path: Path,
) -> "BM25Retriever":
    """Build and save a BM25 index for sparse retrieval.

    Args:
        documents: List of chunked Document objects
        save_path: Path to save the BM25 index

    Returns:
        BM25Retriever instance
    """
    from rag.sparse_retrieval import build_bm25_index as _build_bm25

    return _build_bm25(documents, save_path=save_path)
