"""
Retrieval Configuration for Advanced RAG (Practice 3)

Provides dataclasses for configuring advanced retrieval features:
- Re-ranking (cross-encoder two-stage retrieval)
- Metadata filtering (page range, section)
- Hybrid retrieval (dense + sparse with RRF fusion)

All features are optional and can be enabled/disabled independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass(frozen=True)
class RerankerConfig:
    """Configuration for cross-encoder re-ranking.

    Two-stage retrieval:
    1. Retrieve top_n candidates using dense/hybrid retrieval
    2. Re-rank using cross-encoder, return top_k results

    References:
    - Pinecone: Advanced RAG Techniques (re-ranking)
    - HuggingFace: sentence-transformers cross-encoders
    """
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, good quality
    top_n: int = 20  # Candidates to retrieve before re-ranking
    top_k: int = 5   # Final results after re-ranking


@dataclass(frozen=True)
class MetadataFilter:
    """Configuration for metadata-based filtering.

    Allows restricting retrieval by:
    - Page range (e.g., pages 10-20)
    - Section (e.g., "Financial Highlights")
    - Content type (e.g., "text", "table")

    References:
    - Qdrant: Payload filtering
    - LangChain: Retrieval with metadata
    """
    page_start: Optional[int] = None  # Inclusive
    page_end: Optional[int] = None    # Inclusive
    sections: Optional[List[str]] = None  # Filter by section names
    content_types: Optional[List[str]] = None  # "text", "table", "header"


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for hybrid (dense + sparse) retrieval.

    Combines dense retrieval (embeddings) with sparse retrieval (BM25)
    using Reciprocal Rank Fusion (RRF) for score combination.

    References:
    - Qdrant: Hybrid Search Tutorial
    - Pinecone: Hybrid Search
    - RRF Paper: "Reciprocal Rank Fusion outperforms Condorcet..."
    """
    enabled: bool = False
    sparse_weight: float = 0.3  # Weight for BM25 scores (0.0-1.0)
    dense_weight: float = 0.7   # Weight for embedding scores
    rrf_k: int = 60  # RRF constant (higher = more emphasis on top ranks)


@dataclass
class RetrievalConfig:
    """Complete retrieval configuration.

    Combines all advanced retrieval settings into a single config object.
    Default configuration uses baseline dense retrieval (Practice 1 behavior).
    """
    # Retrieval mode
    mode: Literal["dense", "sparse", "hybrid"] = "dense"

    # Base retrieval settings
    top_k: int = 5  # Final number of documents to return

    # Re-ranking (optional, works with any mode)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    # Metadata filtering (optional)
    metadata_filter: Optional[MetadataFilter] = None

    # Hybrid retrieval settings (only used when mode="hybrid")
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    # Debug/analysis options
    preserve_scores: bool = True  # Keep original scores for analysis

    def with_reranking(
        self,
        top_n: int = 20,
        top_k: int = 5,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> "RetrievalConfig":
        """Create a new config with re-ranking enabled."""
        return RetrievalConfig(
            mode=self.mode,
            top_k=top_k,
            reranker=RerankerConfig(enabled=True, model=model, top_n=top_n, top_k=top_k),
            metadata_filter=self.metadata_filter,
            hybrid=self.hybrid,
            preserve_scores=self.preserve_scores,
        )

    def with_metadata_filter(
        self,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        sections: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
    ) -> "RetrievalConfig":
        """Create a new config with metadata filtering."""
        return RetrievalConfig(
            mode=self.mode,
            top_k=self.top_k,
            reranker=self.reranker,
            metadata_filter=MetadataFilter(
                page_start=page_start,
                page_end=page_end,
                sections=sections,
                content_types=content_types,
            ),
            hybrid=self.hybrid,
            preserve_scores=self.preserve_scores,
        )

    def with_hybrid(
        self,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
    ) -> "RetrievalConfig":
        """Create a new config with hybrid retrieval enabled."""
        return RetrievalConfig(
            mode="hybrid",
            top_k=self.top_k,
            reranker=self.reranker,
            metadata_filter=self.metadata_filter,
            hybrid=HybridConfig(
                enabled=True,
                sparse_weight=sparse_weight,
                dense_weight=dense_weight,
                rrf_k=rrf_k,
            ),
            preserve_scores=self.preserve_scores,
        )


# Preset configurations for common use cases
BASELINE_CONFIG = RetrievalConfig()  # Practice 1 behavior

RERANKED_CONFIG = RetrievalConfig().with_reranking(top_n=20, top_k=5)

HYBRID_CONFIG = RetrievalConfig().with_hybrid(sparse_weight=0.3, dense_weight=0.7)

HYBRID_RERANKED_CONFIG = (
    RetrievalConfig()
    .with_hybrid(sparse_weight=0.3, dense_weight=0.7)
    .with_reranking(top_n=20, top_k=5)
)
