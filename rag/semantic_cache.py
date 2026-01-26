"""
Semantic Caching for RAG System (Practice 4)

Caches query-answer pairs using embeddings-based similarity lookup.
When a semantically similar question is asked, returns cached answer
instead of re-running retrieval and generation.

Features:
- Embeddings-based cache lookup
- Configurable similarity threshold
- Cache hit/miss observability
- Optional enable/disable
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class CacheConfig:
    """Configuration for semantic cache."""
    enabled: bool = True
    similarity_threshold: float = 0.92  # Minimum similarity for cache hit
    max_entries: int = 1000  # Maximum cache entries
    ttl_seconds: Optional[int] = None  # Time-to-live (None = no expiry)
    cache_path: Optional[Path] = None  # Path for persistence


@dataclass
class CacheEntry:
    """Single cache entry with query, answer, and metadata."""
    query: str
    query_embedding: List[float]
    answer: str
    retrieved_contexts: List[str]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, ttl_seconds: Optional[int]) -> bool:
        """Check if entry has expired based on TTL."""
        if ttl_seconds is None:
            return False
        return (time.time() - self.timestamp) > ttl_seconds


@dataclass
class CacheResult:
    """Result from cache lookup."""
    hit: bool
    answer: Optional[str] = None
    contexts: Optional[List[str]] = None
    similarity: Optional[float] = None
    cached_query: Optional[str] = None


class SemanticCache:
    """
    Embeddings-based semantic cache for RAG queries.

    Uses cosine similarity between query embeddings to determine
    if a cached answer can be reused.
    """

    def __init__(
        self,
        embeddings_fn,
        config: Optional[CacheConfig] = None,
    ):
        """
        Initialize semantic cache.

        Args:
            embeddings_fn: Function that takes text and returns embedding vector
            config: Cache configuration
        """
        self.embeddings_fn = embeddings_fn
        self.config = config or CacheConfig()
        self.entries: List[CacheEntry] = []
        self._embedding_matrix: Optional[np.ndarray] = None

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0,
        }

        # Load from disk if path specified
        if self.config.cache_path and self.config.cache_path.exists():
            self._load_cache()

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text."""
        return self.embeddings_fn(text)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _rebuild_embedding_matrix(self):
        """Rebuild the embedding matrix for efficient similarity search."""
        if not self.entries:
            self._embedding_matrix = None
            return

        self._embedding_matrix = np.array([
            entry.query_embedding for entry in self.entries
        ])

    def lookup(self, query: str) -> CacheResult:
        """
        Look up a query in the cache.

        Args:
            query: The query to look up

        Returns:
            CacheResult with hit status and cached data if found
        """
        self.stats["total_queries"] += 1

        if not self.config.enabled or not self.entries:
            self.stats["misses"] += 1
            return CacheResult(hit=False)

        # Compute query embedding
        query_embedding = np.array(self._compute_embedding(query))

        # Clean expired entries first
        if self.config.ttl_seconds:
            self._clean_expired()

        if not self.entries:
            self.stats["misses"] += 1
            return CacheResult(hit=False)

        # Find most similar cached query
        best_similarity = 0.0
        best_entry: Optional[CacheEntry] = None

        if self._embedding_matrix is not None:
            # Vectorized similarity computation
            norms = np.linalg.norm(self._embedding_matrix, axis=1)
            query_norm = np.linalg.norm(query_embedding)

            if query_norm > 0:
                similarities = np.dot(self._embedding_matrix, query_embedding) / (norms * query_norm + 1e-8)
                best_idx = np.argmax(similarities)
                best_similarity = float(similarities[best_idx])
                best_entry = self.entries[best_idx]
        else:
            # Fallback to iterative search
            for entry in self.entries:
                similarity = self._cosine_similarity(
                    query_embedding,
                    np.array(entry.query_embedding)
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

        # Check if similarity exceeds threshold
        if best_entry and best_similarity >= self.config.similarity_threshold:
            self.stats["hits"] += 1
            return CacheResult(
                hit=True,
                answer=best_entry.answer,
                contexts=best_entry.retrieved_contexts,
                similarity=best_similarity,
                cached_query=best_entry.query,
            )

        self.stats["misses"] += 1
        return CacheResult(hit=False, similarity=best_similarity)

    def store(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store a query-answer pair in the cache.

        Args:
            query: The query that was answered
            answer: The generated answer
            contexts: Retrieved contexts used
            metadata: Optional metadata (retrieval mode, scores, etc.)
        """
        if not self.config.enabled:
            return

        # Compute embedding
        query_embedding = self._compute_embedding(query)

        # Create entry
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            answer=answer,
            retrieved_contexts=contexts,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Enforce max entries
        if len(self.entries) >= self.config.max_entries:
            # Remove oldest entry
            self.entries.pop(0)

        self.entries.append(entry)
        self._rebuild_embedding_matrix()

        # Persist if path specified
        if self.config.cache_path:
            self._save_cache()

    def _clean_expired(self):
        """Remove expired entries."""
        if self.config.ttl_seconds is None:
            return

        original_count = len(self.entries)
        self.entries = [
            e for e in self.entries
            if not e.is_expired(self.config.ttl_seconds)
        ]

        if len(self.entries) != original_count:
            self._rebuild_embedding_matrix()

    def clear(self):
        """Clear all cache entries."""
        self.entries = []
        self._embedding_matrix = None
        self.stats = {"hits": 0, "misses": 0, "total_queries": 0}

        if self.config.cache_path and self.config.cache_path.exists():
            self.config.cache_path.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        if self.stats["total_queries"] > 0:
            hit_rate = self.stats["hits"] / self.stats["total_queries"]

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "entries": len(self.entries),
            "max_entries": self.config.max_entries,
            "threshold": self.config.similarity_threshold,
        }

    def _save_cache(self):
        """Persist cache to disk."""
        if not self.config.cache_path:
            return

        data = {
            "entries": [asdict(e) for e in self.entries],
            "stats": self.stats,
            "saved_at": datetime.now().isoformat(),
        }

        self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_cache(self):
        """Load cache from disk."""
        if not self.config.cache_path or not self.config.cache_path.exists():
            return

        try:
            with self.config.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            self.entries = [
                CacheEntry(**entry) for entry in data.get("entries", [])
            ]
            self.stats = data.get("stats", self.stats)
            self._rebuild_embedding_matrix()

            # Clean expired on load
            if self.config.ttl_seconds:
                self._clean_expired()

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load cache: {e}")
            self.entries = []


def create_semantic_cache(
    embeddings_fn,
    enabled: bool = True,
    threshold: float = 0.92,
    max_entries: int = 1000,
    ttl_seconds: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> SemanticCache:
    """
    Factory function to create a semantic cache.

    Args:
        embeddings_fn: Function to compute embeddings
        enabled: Whether caching is enabled
        threshold: Similarity threshold for cache hits (0.0-1.0)
        max_entries: Maximum number of cached entries
        ttl_seconds: Time-to-live for entries (None = no expiry)
        cache_path: Path for persistence (None = in-memory only)

    Returns:
        Configured SemanticCache instance
    """
    config = CacheConfig(
        enabled=enabled,
        similarity_threshold=threshold,
        max_entries=max_entries,
        ttl_seconds=ttl_seconds,
        cache_path=cache_path,
    )
    return SemanticCache(embeddings_fn=embeddings_fn, config=config)
