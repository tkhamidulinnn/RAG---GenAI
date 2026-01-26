"""
Advanced Retriever - Unified Interface (Practice 3)

Provides a single interface for all retrieval modes:
- Dense (baseline): Embedding-based similarity search
- Sparse: BM25 lexical search
- Hybrid: Dense + Sparse with RRF fusion
- Re-ranked: Any mode + cross-encoder re-ranking

Also supports metadata filtering for all modes.

This module preserves backward compatibility with Practice 1
while enabling advanced retrieval features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from rag.retrieval_config import RetrievalConfig, MetadataFilter
from rag.reranker import CrossEncoderReranker, ScoredDocument
from rag.sparse_retrieval import BM25Retriever, load_bm25_index
from rag.hybrid_retrieval import HybridRetriever, HybridResult


@dataclass
class RetrievalResult:
    """Complete retrieval result with all available metadata."""
    documents: List[Document]
    scores: List[float]
    mode: str  # "dense", "sparse", "hybrid"
    reranked: bool
    metadata_filtered: bool

    # Detailed scoring (for analysis)
    original_scores: Optional[List[float]] = None  # Before re-ranking
    rerank_details: Optional[List[ScoredDocument]] = None
    hybrid_details: Optional[List[HybridResult]] = None


class AdvancedRetriever:
    """Unified retriever supporting multiple modes and configurations.

    Modes:
    - "dense": Vector similarity search (FAISS/Qdrant)
    - "sparse": BM25 lexical search
    - "hybrid": Dense + Sparse with RRF fusion

    Optional features:
    - Re-ranking: Cross-encoder second stage
    - Metadata filtering: Page range, section, content type
    """

    def __init__(
        self,
        dense_retriever=None,  # FAISS or Qdrant vector store
        sparse_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        """Initialize the advanced retriever.

        Args:
            dense_retriever: Vector store (FAISS or QdrantVectorStore)
            sparse_retriever: BM25Retriever instance (optional)
            reranker: CrossEncoderReranker instance (optional, lazy loaded)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self._reranker = reranker
        self._hybrid_retriever: Optional[HybridRetriever] = None

    def _get_reranker(self, model: str) -> CrossEncoderReranker:
        """Get or create reranker with the specified model."""
        if self._reranker is None or self._reranker.model_name != model:
            self._reranker = CrossEncoderReranker(model)
        return self._reranker

    def _get_hybrid_retriever(self, config: RetrievalConfig) -> HybridRetriever:
        """Get or create hybrid retriever."""
        if self.dense_retriever is None:
            raise ValueError("Dense retriever required for hybrid mode")
        if self.sparse_retriever is None:
            raise ValueError("Sparse retriever required for hybrid mode")

        return HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            dense_weight=config.hybrid.dense_weight,
            sparse_weight=config.hybrid.sparse_weight,
            rrf_k=config.hybrid.rrf_k,
        )

    def _apply_metadata_filter(
        self,
        documents: List[Document],
        filter_config: MetadataFilter,
    ) -> List[Document]:
        """Filter documents by metadata criteria."""
        filtered = []

        for doc in documents:
            meta = doc.metadata

            # Page range filter
            if filter_config.page_start is not None:
                page = meta.get("page", 0)
                if page < filter_config.page_start:
                    continue
            if filter_config.page_end is not None:
                page = meta.get("page", float("inf"))
                if page > filter_config.page_end:
                    continue

            # Section filter
            if filter_config.sections is not None:
                section = meta.get("section", "")
                if section not in filter_config.sections:
                    continue

            # Content type filter
            if filter_config.content_types is not None:
                content_type = meta.get("content_type", "text")
                if content_type not in filter_config.content_types:
                    continue

            filtered.append(doc)

        return filtered

    def _dense_search(
        self,
        query: str,
        k: int,
        filter_config: Optional[MetadataFilter] = None,
    ) -> Tuple[List[Document], List[float]]:
        """Perform dense (embedding) search."""
        if self.dense_retriever is None:
            raise ValueError("Dense retriever not configured")

        # Retrieve more candidates if we need to filter
        fetch_k = k * 3 if filter_config else k

        try:
            # Try to get scores
            results = self.dense_retriever.similarity_search_with_score(query, k=fetch_k)
            docs = [doc for doc, _ in results]
            scores = [1 / (1 + score) for _, score in results]  # Convert distance to similarity
        except (AttributeError, NotImplementedError):
            # Fall back to search without scores
            docs = self.dense_retriever.similarity_search(query, k=fetch_k)
            scores = [1.0 / (i + 1) for i in range(len(docs))]  # Pseudo-scores

        # Apply metadata filter
        if filter_config:
            filtered_pairs = [
                (doc, score) for doc, score in zip(docs, scores)
                if self._passes_filter(doc, filter_config)
            ]
            docs = [doc for doc, _ in filtered_pairs[:k]]
            scores = [score for _, score in filtered_pairs[:k]]
        else:
            docs = docs[:k]
            scores = scores[:k]

        return docs, scores

    def _passes_filter(self, doc: Document, filter_config: MetadataFilter) -> bool:
        """Check if a document passes the metadata filter."""
        meta = doc.metadata

        if filter_config.page_start is not None:
            if meta.get("page", 0) < filter_config.page_start:
                return False

        if filter_config.page_end is not None:
            if meta.get("page", float("inf")) > filter_config.page_end:
                return False

        if filter_config.sections is not None:
            if meta.get("section", "") not in filter_config.sections:
                return False

        if filter_config.content_types is not None:
            if meta.get("content_type", "text") not in filter_config.content_types:
                return False

        return True

    def _sparse_search(
        self,
        query: str,
        k: int,
        filter_config: Optional[MetadataFilter] = None,
    ) -> Tuple[List[Document], List[float]]:
        """Perform sparse (BM25) search."""
        if self.sparse_retriever is None:
            raise ValueError("Sparse retriever not configured")

        fetch_k = k * 3 if filter_config else k
        results = self.sparse_retriever.search(query, k=fetch_k)

        docs = [doc for doc, _ in results]
        scores = [score for _, score in results]

        if filter_config:
            filtered_pairs = [
                (doc, score) for doc, score in zip(docs, scores)
                if self._passes_filter(doc, filter_config)
            ]
            docs = [doc for doc, _ in filtered_pairs[:k]]
            scores = [score for _, score in filtered_pairs[:k]]
        else:
            docs = docs[:k]
            scores = scores[:k]

        return docs, scores

    def _hybrid_search(
        self,
        query: str,
        k: int,
        config: RetrievalConfig,
    ) -> Tuple[List[Document], List[float], List[HybridResult]]:
        """Perform hybrid (dense + sparse) search."""
        hybrid = self._get_hybrid_retriever(config)

        # Get more candidates if filtering
        fetch_k = k * 3 if config.metadata_filter else k

        results = hybrid.search(query, k=fetch_k)

        if config.metadata_filter:
            results = [
                r for r in results
                if self._passes_filter(r.document, config.metadata_filter)
            ][:k]

        docs = [r.document for r in results]
        scores = [r.final_score for r in results]

        return docs, scores, results

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
    ) -> RetrievalResult:
        """Retrieve documents using the specified configuration.

        This is the main entry point for advanced retrieval.

        Args:
            query: Search query
            config: RetrievalConfig specifying mode and options

        Returns:
            RetrievalResult with documents, scores, and detailed info
        """
        if config is None:
            config = RetrievalConfig()

        # Determine how many candidates to retrieve
        if config.reranker.enabled:
            fetch_k = config.reranker.top_n
            final_k = config.reranker.top_k
        else:
            fetch_k = config.top_k
            final_k = config.top_k

        # Perform search based on mode
        hybrid_details = None
        if config.mode == "dense":
            docs, scores = self._dense_search(query, fetch_k, config.metadata_filter)
        elif config.mode == "sparse":
            docs, scores = self._sparse_search(query, fetch_k, config.metadata_filter)
        elif config.mode == "hybrid":
            docs, scores, hybrid_details = self._hybrid_search(query, fetch_k, config)
        else:
            raise ValueError(f"Unknown retrieval mode: {config.mode}")

        original_scores = scores.copy() if config.preserve_scores else None
        rerank_details = None

        # Apply re-ranking if enabled
        if config.reranker.enabled and docs:
            reranker = self._get_reranker(config.reranker.model)
            scored_docs = reranker.rerank(query, docs, scores, final_k)
            docs = [sd.document for sd in scored_docs]
            scores = [sd.rerank_score or 0.0 for sd in scored_docs]
            rerank_details = scored_docs

        return RetrievalResult(
            documents=docs,
            scores=scores,
            mode=config.mode,
            reranked=config.reranker.enabled,
            metadata_filtered=config.metadata_filter is not None,
            original_scores=original_scores,
            rerank_details=rerank_details,
            hybrid_details=hybrid_details,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """LangChain-compatible interface for baseline retrieval.

        This provides backward compatibility with Practice 1.
        Uses dense retrieval without any advanced features.
        """
        config = RetrievalConfig(mode="dense", top_k=k)
        result = self.retrieve(query, config)
        return result.documents


def create_advanced_retriever(
    dense_retriever=None,
    bm25_index_path: Optional[Path] = None,
    sparse_retriever: Optional[BM25Retriever] = None,
) -> AdvancedRetriever:
    """Create an AdvancedRetriever with optional components.

    Args:
        dense_retriever: Vector store (FAISS or Qdrant)
        bm25_index_path: Path to saved BM25 index
        sparse_retriever: Pre-built BM25Retriever (alternative to path)

    Returns:
        Configured AdvancedRetriever
    """
    # Load BM25 if path provided
    if bm25_index_path and bm25_index_path.exists():
        sparse_retriever = load_bm25_index(bm25_index_path)

    return AdvancedRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )


# Convenience function for backward compatibility
def retrieve(
    db,
    query: str,
    k: int = 5,
    config: Optional[RetrievalConfig] = None,
) -> List[Document]:
    """Drop-in replacement for rag_pipeline.retrieve with optional advanced config.

    Args:
        db: Vector store or AdvancedRetriever
        query: Search query
        k: Number of documents to retrieve
        config: Optional RetrievalConfig for advanced features

    Returns:
        List of Document objects
    """
    if isinstance(db, AdvancedRetriever):
        if config is None:
            config = RetrievalConfig(mode="dense", top_k=k)
        else:
            config = RetrievalConfig(
                mode=config.mode,
                top_k=k,
                reranker=config.reranker,
                metadata_filter=config.metadata_filter,
                hybrid=config.hybrid,
                preserve_scores=config.preserve_scores,
            )
        return db.retrieve(query, config).documents

    # Fall back to simple similarity search for regular vector stores
    return db.similarity_search(query, k=k)
