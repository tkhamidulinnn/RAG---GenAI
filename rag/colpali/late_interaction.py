"""
Late Interaction Retrieval for ColPali-style RAG (Phase 6)

Implements MaxSim-style scoring where query embeddings are compared
against all patch embeddings, then aggregated to rank pages/regions.

Key concepts:
- Late interaction: Compare query against patches at retrieval time
- MaxSim: For each query token, find max similarity across patches
- Page aggregation: Combine patch scores to rank full pages
- Region highlighting: Identify which patches are most relevant
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from rag.colpali.patch_embeddings import PatchEmbedding, load_embeddings, load_faiss_index


@dataclass
class PatchScore:
    """Score for a single patch."""
    patch_id: str
    page: int
    doc_id: str
    score: float
    # Location for visualization
    x: float
    y: float
    width: float
    height: float


@dataclass
class PageScore:
    """Aggregated score for a page."""
    page: int
    doc_id: str
    score: float
    top_patches: List[PatchScore]
    # Statistics
    mean_patch_score: float
    max_patch_score: float
    num_relevant_patches: int


@dataclass
class RetrievalResult:
    """Result of late interaction retrieval."""
    query: str
    query_embedding: List[float]
    # Results ranked by score
    page_scores: List[PageScore]
    patch_scores: List[PatchScore]
    # Retrieval metadata
    total_patches_searched: int
    total_pages: int
    retrieval_method: str = "late_interaction"


def compute_similarity(
    query_embedding: np.ndarray,
    patch_embeddings: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute cosine similarity between query and all patches.

    Args:
        query_embedding: Query vector (d,)
        patch_embeddings: Patch matrix (n_patches, d)
        normalize: Whether to L2 normalize before dot product

    Returns:
        Similarity scores (n_patches,)
    """
    if normalize:
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        patch_norms = patch_embeddings / (np.linalg.norm(patch_embeddings, axis=1, keepdims=True) + 1e-9)
        return np.dot(patch_norms, query_norm)
    return np.dot(patch_embeddings, query_embedding)


def late_interaction_search(
    query_embedding: List[float],
    patch_embeddings: List[PatchEmbedding],
    embeddings_array: Optional[np.ndarray] = None,
    top_k_patches: int = 20,
    score_threshold: float = 0.0,
) -> List[PatchScore]:
    """
    Perform late interaction search over patch embeddings.

    This is the core MaxSim-style operation:
    - Compare query against all patches
    - Return top-k highest scoring patches

    Args:
        query_embedding: Query vector
        patch_embeddings: List of PatchEmbedding objects (for metadata)
        embeddings_array: Pre-loaded numpy array of embeddings (faster)
        top_k_patches: Number of top patches to return
        score_threshold: Minimum score to include

    Returns:
        List of PatchScore objects, sorted by score descending
    """
    if not patch_embeddings:
        return []

    # Convert to numpy
    query_vec = np.array(query_embedding, dtype=np.float32)

    if embeddings_array is not None:
        patch_matrix = embeddings_array
    else:
        patch_matrix = np.array([p.embedding for p in patch_embeddings], dtype=np.float32)

    # Compute similarities
    scores = compute_similarity(query_vec, patch_matrix)

    # Get top-k indices
    if top_k_patches >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        # Partial sort for efficiency
        top_indices = np.argpartition(scores, -top_k_patches)[-top_k_patches:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    # Build results
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < score_threshold:
            continue

        patch = patch_embeddings[idx]
        results.append(PatchScore(
            patch_id=patch.patch_id,
            page=patch.page,
            doc_id=patch.doc_id,
            score=score,
            x=patch.x,
            y=patch.y,
            width=patch.width,
            height=patch.height,
        ))

    return results


def late_interaction_search_faiss(
    query_embedding: List[float],
    index,  # faiss.Index
    patch_embeddings: List[PatchEmbedding],
    top_k_patches: int = 20,
) -> List[PatchScore]:
    """
    Perform late interaction search using FAISS index.

    Faster for large collections but less flexible for scoring.

    Args:
        query_embedding: Query vector
        index: FAISS index (IndexFlatIP for cosine similarity)
        patch_embeddings: List of PatchEmbedding objects (for metadata)
        top_k_patches: Number of top patches to return

    Returns:
        List of PatchScore objects
    """
    import faiss

    # Prepare query
    query_vec = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vec)

    # Search
    scores, indices = index.search(query_vec, min(top_k_patches, index.ntotal))

    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for invalid
            continue

        patch = patch_embeddings[idx]
        results.append(PatchScore(
            patch_id=patch.patch_id,
            page=patch.page,
            doc_id=patch.doc_id,
            score=float(score),
            x=patch.x,
            y=patch.y,
            width=patch.width,
            height=patch.height,
        ))

    return results


def aggregate_page_scores(
    patch_scores: List[PatchScore],
    aggregation: str = "max",
    top_patches_per_page: int = 3,
) -> List[PageScore]:
    """
    Aggregate patch scores to page level.

    Args:
        patch_scores: List of scored patches
        aggregation: How to combine scores ("max", "mean", "sum", "weighted")
        top_patches_per_page: Number of top patches to include per page

    Returns:
        List of PageScore objects, sorted by score descending
    """
    if not patch_scores:
        return []

    # Group patches by (doc_id, page)
    page_patches: Dict[Tuple[str, int], List[PatchScore]] = {}
    for ps in patch_scores:
        key = (ps.doc_id, ps.page)
        if key not in page_patches:
            page_patches[key] = []
        page_patches[key].append(ps)

    # Compute page scores
    page_results = []
    for (doc_id, page), patches in page_patches.items():
        scores = [p.score for p in patches]

        # Aggregation method
        if aggregation == "max":
            agg_score = max(scores)
        elif aggregation == "mean":
            agg_score = np.mean(scores)
        elif aggregation == "sum":
            agg_score = sum(scores)
        elif aggregation == "weighted":
            # Weight by patch position (top patches more important)
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            sorted_scores = sorted(scores, reverse=True)
            agg_score = sum(s * w for s, w in zip(sorted_scores, weights))
        else:
            agg_score = max(scores)

        # Get top patches for this page
        top_patches = sorted(patches, key=lambda p: p.score, reverse=True)[:top_patches_per_page]

        # Count relevant patches (score > 0.5 of max)
        max_score = max(scores)
        threshold = max_score * 0.5
        num_relevant = sum(1 for s in scores if s >= threshold)

        page_results.append(PageScore(
            page=page,
            doc_id=doc_id,
            score=agg_score,
            top_patches=top_patches,
            mean_patch_score=float(np.mean(scores)),
            max_patch_score=max_score,
            num_relevant_patches=num_relevant,
        ))

    # Sort by score
    return sorted(page_results, key=lambda p: p.score, reverse=True)


def retrieve_with_late_interaction(
    query: str,
    query_embedding: List[float],
    embeddings_dir,
    top_k_pages: int = 5,
    top_k_patches: int = 50,
    aggregation: str = "max",
    use_faiss: bool = True,
    verbose: bool = False,
) -> RetrievalResult:
    """
    Full late interaction retrieval pipeline.

    Args:
        query: Original query text
        query_embedding: Query embedding vector
        embeddings_dir: Directory containing embeddings and index
        top_k_pages: Number of top pages to return
        top_k_patches: Number of patches to consider
        aggregation: Score aggregation method
        use_faiss: Use FAISS index if available
        verbose: Print debug info

    Returns:
        RetrievalResult with ranked pages and patches
    """
    from pathlib import Path
    embeddings_dir = Path(embeddings_dir)

    # Load embeddings
    patch_embeddings, emb_array = load_embeddings(embeddings_dir)

    if not patch_embeddings:
        return RetrievalResult(
            query=query,
            query_embedding=query_embedding,
            page_scores=[],
            patch_scores=[],
            total_patches_searched=0,
            total_pages=0,
        )

    # Get unique pages
    unique_pages = set((p.doc_id, p.page) for p in patch_embeddings)

    if verbose:
        print(f"  Searching {len(patch_embeddings)} patches across {len(unique_pages)} pages...")

    # Perform search
    if use_faiss:
        index = load_faiss_index(embeddings_dir)
        if index is not None:
            patch_scores = late_interaction_search_faiss(
                query_embedding, index, patch_embeddings, top_k_patches
            )
        else:
            patch_scores = late_interaction_search(
                query_embedding, patch_embeddings, emb_array, top_k_patches
            )
    else:
        patch_scores = late_interaction_search(
            query_embedding, patch_embeddings, emb_array, top_k_patches
        )

    if verbose:
        print(f"  Found {len(patch_scores)} relevant patches")

    # Aggregate to page level
    page_scores = aggregate_page_scores(patch_scores, aggregation)[:top_k_pages]

    if verbose:
        print(f"  Top pages: {[p.page for p in page_scores]}")

    return RetrievalResult(
        query=query,
        query_embedding=query_embedding,
        page_scores=page_scores,
        patch_scores=patch_scores,
        total_patches_searched=len(patch_embeddings),
        total_pages=len(unique_pages),
    )


class LateInteractionRetriever:
    """
    Retriever class for late interaction / MaxSim-style retrieval.

    Encapsulates the retrieval pipeline with pre-loaded resources.
    """

    def __init__(
        self,
        embeddings_dir,
        client=None,
        use_clip: bool = False,
        use_faiss: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize retriever with embeddings.

        Args:
            embeddings_dir: Directory containing patch embeddings
            client: Gemini client for query embedding
            use_clip: Use CLIP for query embedding
            use_faiss: Use FAISS index for search
            verbose: Print debug info
        """
        from pathlib import Path
        self.embeddings_dir = Path(embeddings_dir)
        self.client = client
        self.use_clip = use_clip
        self.use_faiss = use_faiss
        self.verbose = verbose

        # Load resources
        self.patch_embeddings, self.emb_array = load_embeddings(self.embeddings_dir)

        if use_faiss:
            self.index = load_faiss_index(self.embeddings_dir)
        else:
            self.index = None

        if verbose:
            print(f"LateInteractionRetriever: Loaded {len(self.patch_embeddings)} patches")

    def retrieve(
        self,
        query: str,
        top_k_pages: int = 5,
        top_k_patches: int = 50,
        aggregation: str = "max",
    ) -> RetrievalResult:
        """
        Retrieve relevant pages for a query.

        Args:
            query: Text query
            top_k_pages: Number of pages to return
            top_k_patches: Number of patches to consider
            aggregation: Score aggregation method

        Returns:
            RetrievalResult with ranked pages
        """
        from rag.colpali.patch_embeddings import generate_query_embedding

        # Generate query embedding
        query_embedding = generate_query_embedding(
            query, self.client, self.use_clip
        )

        if not self.patch_embeddings:
            return RetrievalResult(
                query=query,
                query_embedding=query_embedding,
                page_scores=[],
                patch_scores=[],
                total_patches_searched=0,
                total_pages=0,
            )

        # Perform search
        if self.index is not None:
            patch_scores = late_interaction_search_faiss(
                query_embedding, self.index, self.patch_embeddings, top_k_patches
            )
        else:
            patch_scores = late_interaction_search(
                query_embedding, self.patch_embeddings, self.emb_array, top_k_patches
            )

        # Aggregate
        unique_pages = set((p.doc_id, p.page) for p in self.patch_embeddings)
        page_scores = aggregate_page_scores(patch_scores, aggregation)[:top_k_pages]

        return RetrievalResult(
            query=query,
            query_embedding=query_embedding,
            page_scores=page_scores,
            patch_scores=patch_scores,
            total_patches_searched=len(self.patch_embeddings),
            total_pages=len(unique_pages),
        )

    def get_page_patches(self, doc_id: str, page: int) -> List[PatchEmbedding]:
        """Get all patches for a specific page."""
        return [
            p for p in self.patch_embeddings
            if p.doc_id == doc_id and p.page == page
        ]

    def get_patch_by_id(self, patch_id: str) -> Optional[PatchEmbedding]:
        """Get a specific patch by ID."""
        for p in self.patch_embeddings:
            if p.patch_id == patch_id:
                return p
        return None
