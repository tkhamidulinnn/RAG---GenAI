"""
Hybrid Retrieval with Reciprocal Rank Fusion (Practice 3)

Combines dense (embedding) and sparse (BM25) retrieval for improved
search quality. Uses Reciprocal Rank Fusion (RRF) to merge results
from multiple retrieval methods.

Why hybrid retrieval?
- Dense: Good semantic understanding, handles paraphrasing
- Sparse: Good exact matching, handles rare terms, technical vocabulary
- Hybrid: Best of both worlds

References:
- Qdrant: Hybrid Search Tutorial
- Pinecone: Hybrid Search
- RRF Paper: "Reciprocal Rank Fusion outperforms Condorcet and individual
  Rank Learning Methods" (Cormack et al., 2009)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document


@dataclass
class HybridResult:
    """Result from hybrid retrieval with detailed scoring."""
    document: Document
    final_score: float
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[Document, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ weight_i / (k + rank_i(d))

    This method is robust and doesn't require score normalization.
    Higher k values give more emphasis to top-ranked documents.

    Args:
        ranked_lists: List of ranked result lists, each containing (Document, score) tuples
        k: RRF constant (default: 60, as in the original paper)
        weights: Optional weights for each list (default: equal weights)

    Returns:
        Merged list of (Document, score) tuples, sorted by RRF score
    """
    if not ranked_lists:
        return []

    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("Number of weights must match number of ranked lists")

    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    # Compute RRF scores
    # Use page_content as key since Document objects might not be hashable
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        weight = weights[list_idx]
        for rank, (doc, _) in enumerate(ranked_list, start=1):
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0) + weight / (k + rank)

    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[key], score) for key, score in sorted_results]


def hybrid_search(
    query: str,
    dense_retriever,  # Vector store with similarity_search_with_score
    sparse_retriever,  # BM25Retriever
    top_k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    candidates_per_retriever: int = 20,
    rrf_k: int = 60,
) -> List[HybridResult]:
    """Perform hybrid search combining dense and sparse retrieval.

    Two-stage process:
    1. Retrieve candidates from both dense and sparse retrievers
    2. Merge results using RRF with configurable weights

    Args:
        query: Search query
        dense_retriever: Vector store (FAISS or Qdrant)
        sparse_retriever: BM25Retriever instance
        top_k: Number of final results to return
        dense_weight: Weight for dense retrieval (0.0-1.0)
        sparse_weight: Weight for sparse retrieval (0.0-1.0)
        candidates_per_retriever: Number of candidates from each retriever
        rrf_k: RRF constant

    Returns:
        List of HybridResult with detailed scoring information
    """
    # Get dense results
    dense_results = []
    try:
        # Try similarity_search_with_score first
        dense_raw = dense_retriever.similarity_search_with_score(
            query, k=candidates_per_retriever
        )
        # FAISS returns (doc, distance), convert to (doc, similarity)
        # Lower distance = higher similarity
        for doc, score in dense_raw:
            # Normalize: convert distance to similarity (1 / (1 + distance))
            similarity = 1 / (1 + score) if score >= 0 else 0
            dense_results.append((doc, similarity))
    except (AttributeError, NotImplementedError):
        # Fall back to similarity_search without scores
        docs = dense_retriever.similarity_search(query, k=candidates_per_retriever)
        # Assign decreasing pseudo-scores based on rank
        dense_results = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]

    # Get sparse (BM25) results
    sparse_results = sparse_retriever.search(query, k=candidates_per_retriever)

    # Apply RRF fusion
    merged = reciprocal_rank_fusion(
        ranked_lists=[dense_results, sparse_results],
        k=rrf_k,
        weights=[dense_weight, sparse_weight],
    )

    # Build detailed results with per-retriever info
    dense_rank_map = {doc.page_content: (i + 1, score) for i, (doc, score) in enumerate(dense_results)}
    sparse_rank_map = {doc.page_content: (i + 1, score) for i, (doc, score) in enumerate(sparse_results)}

    results = []
    for doc, rrf_score in merged[:top_k]:
        key = doc.page_content
        dense_info = dense_rank_map.get(key)
        sparse_info = sparse_rank_map.get(key)

        results.append(
            HybridResult(
                document=doc,
                final_score=rrf_score,
                dense_rank=dense_info[0] if dense_info else None,
                dense_score=dense_info[1] if dense_info else None,
                sparse_rank=sparse_info[0] if sparse_info else None,
                sparse_score=sparse_info[1] if sparse_info else None,
            )
        )

    return results


def hybrid_search_documents(
    query: str,
    dense_retriever,
    sparse_retriever,
    top_k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    candidates_per_retriever: int = 20,
    rrf_k: int = 60,
) -> List[Document]:
    """Convenience function that returns only documents.

    Args:
        Same as hybrid_search

    Returns:
        List of Document objects (without detailed scores)
    """
    results = hybrid_search(
        query=query,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k=top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        candidates_per_retriever=candidates_per_retriever,
        rrf_k=rrf_k,
    )
    return [r.document for r in results]


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search.

    Encapsulates both retrievers and provides a unified interface.
    """

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        candidates_per_retriever: int = 20,
        rrf_k: int = 60,
    ):
        """Initialize hybrid retriever.

        Args:
            dense_retriever: Vector store (FAISS or Qdrant)
            sparse_retriever: BM25Retriever instance
            dense_weight: Weight for dense results
            sparse_weight: Weight for sparse results
            candidates_per_retriever: Candidates to retrieve before fusion
            rrf_k: RRF constant
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidates_per_retriever = candidates_per_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> List[HybridResult]:
        """Search with detailed results."""
        return hybrid_search(
            query=query,
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            top_k=k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            candidates_per_retriever=self.candidates_per_retriever,
            rrf_k=self.rrf_k,
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """LangChain-compatible interface returning documents only."""
        results = self.search(query, k)
        return [r.document for r in results]
