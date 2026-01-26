"""
Sparse Retrieval with BM25 (Practice 3)

Implements BM25-based sparse retrieval as an alternative/complement
to dense (embedding-based) retrieval.

BM25 is particularly effective for:
- Exact keyword matching
- Technical terminology
- Rare terms that embeddings might not capture well

References:
- Qdrant: Hybrid Search & Re-ranking
- Pinecone: Hybrid Search
- LangChain: BM25Retriever
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

from langchain_core.documents import Document


@dataclass
class BM25Index:
    """BM25 index for sparse retrieval."""
    documents: List[Document]
    doc_freqs: Dict[str, int]  # Document frequency for each term
    doc_lengths: List[int]  # Length of each document
    avg_doc_length: float
    total_docs: int
    k1: float = 1.5  # Term frequency saturation parameter
    b: float = 0.75  # Length normalization parameter


class BM25Retriever:
    """BM25-based sparse retriever.

    Implements the BM25 (Best Matching 25) algorithm for lexical retrieval.
    Unlike dense retrieval, BM25 uses exact term matching and TF-IDF weighting.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
                Higher values give more weight to term frequency.
            b: Length normalization parameter (default: 0.75)
                0 = no normalization, 1 = full normalization
        """
        self.k1 = k1
        self.b = b
        self._index: Optional[BM25Index] = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms.

        Simple whitespace tokenization with lowercasing and
        punctuation removal. Production systems might use
        more sophisticated tokenization.
        """
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)

        # Split on whitespace and filter empty/short tokens
        tokens = text.split()
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def index_documents(self, documents: List[Document]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of LangChain Document objects
        """
        doc_freqs: Dict[str, int] = Counter()
        doc_lengths: List[int] = []
        total_length = 0

        # First pass: compute document frequencies and lengths
        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            doc_lengths.append(len(tokens))
            total_length += len(tokens)

            # Count unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freqs[term] += 1

        avg_doc_length = total_length / len(documents) if documents else 0

        self._index = BM25Index(
            documents=documents,
            doc_freqs=dict(doc_freqs),
            doc_lengths=doc_lengths,
            avg_doc_length=avg_doc_length,
            total_docs=len(documents),
            k1=self.k1,
            b=self.b,
        )

    def _compute_idf(self, term: str) -> float:
        """Compute IDF (inverse document frequency) for a term."""
        if self._index is None:
            return 0.0

        n = self._index.total_docs
        df = self._index.doc_freqs.get(term, 0)

        if df == 0:
            return 0.0

        # Standard IDF formula with smoothing
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _score_document(
        self,
        query_terms: List[str],
        doc_index: int,
    ) -> float:
        """Compute BM25 score for a single document."""
        if self._index is None:
            return 0.0

        doc = self._index.documents[doc_index]
        doc_tokens = self._tokenize(doc.page_content)
        doc_length = self._index.doc_lengths[doc_index]
        avg_length = self._index.avg_doc_length

        # Term frequencies in this document
        tf_map = Counter(doc_tokens)

        score = 0.0
        for term in query_terms:
            if term not in tf_map:
                continue

            tf = tf_map[term]
            idf = self._compute_idf(term)

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avg_length)
            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """Search for documents matching the query.

        Args:
            query: Search query
            k: Number of top documents to return

        Returns:
            List of (Document, score) tuples, sorted by score descending
        """
        if self._index is None:
            raise ValueError("Index not built. Call index_documents() first.")

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Score all documents
        scores = []
        for i in range(len(self._index.documents)):
            score = self._score_document(query_terms, i)
            if score > 0:
                scores.append((i, score))

        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in scores[:k]:
            results.append((self._index.documents[doc_idx], score))

        return results

    def search_documents(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """Search and return only documents (without scores).

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of Document objects
        """
        results = self.search(query, k)
        return [doc for doc, _ in results]

    def save(self, path: Path) -> None:
        """Save BM25 index to disk.

        Args:
            path: Path to save the index (JSON format)
        """
        if self._index is None:
            raise ValueError("No index to save. Call index_documents() first.")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize documents as dicts
        doc_dicts = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in self._index.documents
        ]

        data = {
            "documents": doc_dicts,
            "doc_freqs": self._index.doc_freqs,
            "doc_lengths": self._index.doc_lengths,
            "avg_doc_length": self._index.avg_doc_length,
            "total_docs": self._index.total_docs,
            "k1": self._index.k1,
            "b": self._index.b,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: Path) -> None:
        """Load BM25 index from disk.

        Args:
            path: Path to the saved index
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct documents
        documents = [
            Document(
                page_content=d["page_content"],
                metadata=d["metadata"],
            )
            for d in data["documents"]
        ]

        self._index = BM25Index(
            documents=documents,
            doc_freqs=data["doc_freqs"],
            doc_lengths=data["doc_lengths"],
            avg_doc_length=data["avg_doc_length"],
            total_docs=data["total_docs"],
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
        )

    @property
    def is_indexed(self) -> bool:
        """Check if the index has been built."""
        return self._index is not None


def build_bm25_index(
    documents: List[Document],
    save_path: Optional[Path] = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Retriever:
    """Build a BM25 index from documents.

    Args:
        documents: List of LangChain Document objects
        save_path: Optional path to save the index
        k1: BM25 term frequency saturation parameter
        b: BM25 length normalization parameter

    Returns:
        Initialized BM25Retriever with indexed documents
    """
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.index_documents(documents)

    if save_path:
        retriever.save(save_path)

    return retriever


def load_bm25_index(path: Path) -> BM25Retriever:
    """Load a BM25 index from disk.

    Args:
        path: Path to the saved index

    Returns:
        BM25Retriever with loaded index
    """
    retriever = BM25Retriever()
    retriever.load(path)
    return retriever
