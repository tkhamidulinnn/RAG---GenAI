"""
Cross-Encoder Re-ranking Module (Practice 3)

Implements two-stage retrieval:
1. First stage: Retrieve Top-N candidates using dense/hybrid retrieval
2. Second stage: Re-rank using cross-encoder model, return Top-K

Cross-encoders score (query, document) pairs jointly, providing more accurate
relevance scores than bi-encoders but at higher computational cost.

References:
- Pinecone: Advanced RAG Techniques
- HuggingFace sentence-transformers: Cross-Encoders
- MS MARCO: Cross-encoder training for passage re-ranking
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.documents import Document


@dataclass
class ScoredDocument:
    """Document with both original and re-ranked scores."""
    document: Document
    original_score: float
    rerank_score: Optional[float] = None
    original_rank: int = 0
    rerank_rank: Optional[int] = None


class CrossEncoderReranker:
    """Cross-encoder based re-ranker using sentence-transformers.

    Loads a cross-encoder model that scores (query, passage) pairs.
    Higher scores indicate higher relevance.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the re-ranker with a cross-encoder model.

        Args:
            model_name: HuggingFace model name. Recommended models:
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
                - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest, lower quality)
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, higher quality)
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for re-ranking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[ScoredDocument]:
        """Re-rank documents using cross-encoder scores.

        Args:
            query: The search query
            documents: List of candidate documents to re-rank
            original_scores: Original similarity scores (optional, for analysis)
            top_k: Number of top documents to return after re-ranking

        Returns:
            List of ScoredDocument with re-ranked scores, sorted by relevance
        """
        if not documents:
            return []

        # Prepare (query, passage) pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Get cross-encoder scores
        model = self._load_model()
        scores = model.predict(pairs)

        # Create scored documents with original ranks
        scored_docs = []
        for i, (doc, rerank_score) in enumerate(zip(documents, scores)):
            orig_score = original_scores[i] if original_scores else 0.0
            scored_docs.append(
                ScoredDocument(
                    document=doc,
                    original_score=orig_score,
                    rerank_score=float(rerank_score),
                    original_rank=i + 1,
                )
            )

        # Sort by re-rank score (descending) and assign new ranks
        scored_docs.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        for i, sd in enumerate(scored_docs):
            sd.rerank_rank = i + 1

        # Return top_k
        return scored_docs[:top_k]

    def rerank_with_details(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> Tuple[List[Document], List[ScoredDocument]]:
        """Re-rank and return both documents and detailed scores.

        Useful for evaluation and debugging to see how re-ranking
        changed document order.

        Returns:
            Tuple of (reranked documents, full scored document list with details)
        """
        scored_docs = self.rerank(query, documents, original_scores, top_k)
        return [sd.document for sd in scored_docs], scored_docs


# Global reranker instance (lazy loaded)
_default_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    """Get or create the default reranker instance."""
    global _default_reranker
    if _default_reranker is None or _default_reranker.model_name != model_name:
        _default_reranker = CrossEncoderReranker(model_name)
    return _default_reranker


def rerank_documents(
    query: str,
    documents: List[Document],
    original_scores: Optional[List[float]] = None,
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Document]:
    """Convenience function to re-rank documents.

    Args:
        query: The search query
        documents: Candidate documents from first-stage retrieval
        original_scores: Optional original similarity scores
        top_k: Number of documents to return
        model_name: Cross-encoder model to use

    Returns:
        List of re-ranked documents (top_k)
    """
    reranker = get_reranker(model_name)
    scored_docs = reranker.rerank(query, documents, original_scores, top_k)
    return [sd.document for sd in scored_docs]
