"""
Comparison Mode: Classic RAG vs ColPali Pipeline (Phase 6)

Runs both pipelines in parallel and compares:
- Retrieval results (which documents/regions are found)
- Answer quality (grounding, completeness)
- Visual attribution

This helps evaluate when visual-first retrieval adds value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

from langchain_core.documents import Document


@dataclass
class ClassicRAGResult:
    """Result from classic text-based RAG."""
    query: str
    documents: List[Document]
    answer: str
    retrieval_time_ms: float
    generation_time_ms: float
    # Metadata
    retrieval_mode: str = "dense"
    top_k: int = 5
    reranked: bool = False


@dataclass
class ColPaliResult:
    """Result from ColPali visual pipeline."""
    query: str
    page_scores: List[Dict[str, Any]]
    patch_scores: List[Dict[str, Any]]
    answer: str
    retrieval_time_ms: float
    generation_time_ms: float
    # Attribution
    attribution_summary: str = ""
    highlighted_pages: List[int] = field(default_factory=list)
    # Metadata
    total_patches_searched: int = 0
    aggregation_method: str = "max"


@dataclass
class ComparisonResult:
    """Side-by-side comparison of both pipelines."""
    query: str
    classic: ClassicRAGResult
    colpali: ColPaliResult
    # Comparison metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Recommendations
    preferred_pipeline: str = ""
    reason: str = ""


def run_classic_pipeline(
    query: str,
    retriever,
    client,
    model: str,
    top_k: int = 5,
    rerank: bool = False,
) -> ClassicRAGResult:
    """
    Run classic text-based RAG pipeline.

    Args:
        query: User query
        retriever: LangChain-compatible retriever (FAISS/Qdrant)
        client: Gemini client
        model: Model name
        top_k: Number of documents to retrieve
        rerank: Whether to apply re-ranking

    Returns:
        ClassicRAGResult
    """
    from rag.rag_pipeline import retrieve, answer_stream

    # Retrieval
    start = time.time()

    if hasattr(retriever, "retrieve"):
        # Advanced retriever
        from rag.retrieval_config import RetrievalConfig, RerankerConfig

        config = RetrievalConfig(
            mode="dense",
            top_k=top_k,
            reranker=RerankerConfig(enabled=rerank, top_n=top_k * 2, top_k=top_k) if rerank else None,
        )
        result = retriever.retrieve(query, config)
        docs = result.documents
    else:
        # Simple FAISS/Qdrant retriever
        docs = retrieve(retriever, query, k=top_k)

    retrieval_time = (time.time() - start) * 1000

    # Generation
    start = time.time()
    answer = ""
    for token in answer_stream(client, model, query, docs):
        answer += token
    generation_time = (time.time() - start) * 1000

    return ClassicRAGResult(
        query=query,
        documents=docs,
        answer=answer,
        retrieval_time_ms=retrieval_time,
        generation_time_ms=generation_time,
        retrieval_mode="dense",
        top_k=top_k,
        reranked=rerank,
    )


def run_colpali_pipeline(
    query: str,
    embeddings_dir: Path,
    pages_dir: Path,
    client,
    model: str,
    top_k_pages: int = 5,
    top_k_patches: int = 50,
    use_clip: bool = False,
) -> ColPaliResult:
    """
    Run ColPali visual pipeline.

    Args:
        query: User query
        embeddings_dir: Directory with patch embeddings
        pages_dir: Directory with page images
        client: Gemini client
        model: Model name
        top_k_pages: Number of pages to return
        top_k_patches: Number of patches to consider
        use_clip: Use CLIP embeddings

    Returns:
        ColPaliResult
    """
    from rag.colpali.late_interaction import LateInteractionRetriever
    from rag.colpali.visual_generation import collect_visual_context, generate_visual_answer
    from rag.colpali.source_attribution import generate_attribution

    # Retrieval
    start = time.time()

    retriever = LateInteractionRetriever(
        embeddings_dir=embeddings_dir,
        client=client,
        use_clip=use_clip,
    )

    result = retriever.retrieve(
        query=query,
        top_k_pages=top_k_pages,
        top_k_patches=top_k_patches,
    )

    retrieval_time = (time.time() - start) * 1000

    # Convert to dicts for serialization
    page_scores = [
        {
            "page": ps.page,
            "doc_id": ps.doc_id,
            "score": ps.score,
            "mean_patch_score": ps.mean_patch_score,
            "max_patch_score": ps.max_patch_score,
            "num_relevant_patches": ps.num_relevant_patches,
        }
        for ps in result.page_scores
    ]

    patch_scores = [
        {
            "patch_id": ps.patch_id,
            "page": ps.page,
            "score": ps.score,
            "x": ps.x,
            "y": ps.y,
        }
        for ps in result.patch_scores[:20]  # Limit for display
    ]

    # Generation
    start = time.time()

    contexts = collect_visual_context(result, pages_dir, max_pages=top_k_pages)
    visual_answer = generate_visual_answer(client, model, query, contexts)
    answer = visual_answer.answer

    generation_time = (time.time() - start) * 1000

    # Attribution
    attribution = generate_attribution(
        result, pages_dir, max_pages=top_k_pages, create_visualizations=False
    )

    return ColPaliResult(
        query=query,
        page_scores=page_scores,
        patch_scores=patch_scores,
        answer=answer,
        retrieval_time_ms=retrieval_time,
        generation_time_ms=generation_time,
        attribution_summary=attribution.summary,
        highlighted_pages=[ps["page"] for ps in page_scores],
        total_patches_searched=result.total_patches_searched,
    )


def compare_results(
    classic: ClassicRAGResult,
    colpali: ColPaliResult,
) -> ComparisonResult:
    """
    Compare classic and ColPali results.

    Args:
        classic: Result from classic pipeline
        colpali: Result from ColPali pipeline

    Returns:
        ComparisonResult with metrics and recommendation
    """
    metrics = {}

    # Timing comparison
    metrics["classic_retrieval_ms"] = classic.retrieval_time_ms
    metrics["colpali_retrieval_ms"] = colpali.retrieval_time_ms
    metrics["classic_generation_ms"] = classic.generation_time_ms
    metrics["colpali_generation_ms"] = colpali.generation_time_ms

    metrics["retrieval_speedup"] = classic.retrieval_time_ms / max(colpali.retrieval_time_ms, 0.1)
    metrics["total_classic_ms"] = classic.retrieval_time_ms + classic.generation_time_ms
    metrics["total_colpali_ms"] = colpali.retrieval_time_ms + colpali.generation_time_ms

    # Coverage comparison
    classic_pages = set(d.metadata.get("page") for d in classic.documents if d.metadata.get("page"))
    colpali_pages = set(ps["page"] for ps in colpali.page_scores)

    metrics["classic_pages"] = sorted(classic_pages)
    metrics["colpali_pages"] = sorted(colpali_pages)
    metrics["page_overlap"] = len(classic_pages & colpali_pages)
    metrics["classic_only_pages"] = sorted(classic_pages - colpali_pages)
    metrics["colpali_only_pages"] = sorted(colpali_pages - classic_pages)

    # Answer comparison
    classic_words = set(classic.answer.lower().split())
    colpali_words = set(colpali.answer.lower().split())

    metrics["classic_answer_length"] = len(classic.answer)
    metrics["colpali_answer_length"] = len(colpali.answer)
    metrics["answer_word_overlap"] = len(classic_words & colpali_words)
    metrics["answer_similarity"] = (
        len(classic_words & colpali_words) / max(len(classic_words | colpali_words), 1)
    )

    # Determine preference
    preferred, reason = determine_preference(classic, colpali, metrics)

    return ComparisonResult(
        query=classic.query,
        classic=classic,
        colpali=colpali,
        metrics=metrics,
        preferred_pipeline=preferred,
        reason=reason,
    )


def determine_preference(
    classic: ClassicRAGResult,
    colpali: ColPaliResult,
    metrics: Dict[str, Any],
) -> tuple[str, str]:
    """
    Determine which pipeline is preferred for this query.

    Heuristics:
    - If ColPali finds different pages with high scores → visual might be better
    - If answers are similar → use faster one
    - If query mentions visual elements → prefer ColPali
    """
    query_lower = classic.query.lower()

    # Check for visual keywords
    visual_keywords = [
        "chart", "graph", "figure", "table", "diagram", "image",
        "visual", "show", "display", "illustration", "picture",
    ]
    has_visual_intent = any(kw in query_lower for kw in visual_keywords)

    # Check page overlap
    low_overlap = metrics["page_overlap"] < min(len(metrics["classic_pages"]), len(metrics["colpali_pages"])) / 2

    # Check ColPali found unique content
    colpali_unique = len(metrics["colpali_only_pages"]) > 0

    # Decision logic
    if has_visual_intent:
        return "colpali", "Query has visual intent - ColPali uses visual understanding"

    if low_overlap and colpali_unique:
        return "colpali", "ColPali found different relevant pages - may capture visual layout"

    if metrics["answer_similarity"] > 0.7:
        # Similar answers - prefer faster
        if metrics["total_classic_ms"] < metrics["total_colpali_ms"]:
            return "classic", "Similar answers - classic is faster"
        else:
            return "colpali", "Similar answers - ColPali provides visual attribution"

    if metrics["colpali_answer_length"] > metrics["classic_answer_length"] * 1.5:
        return "colpali", "ColPali provides more detailed answer"

    return "classic", "Classic RAG is more efficient for this text-based query"


def run_comparison(
    query: str,
    classic_retriever,
    embeddings_dir: Path,
    pages_dir: Path,
    client,
    model: str,
    classic_top_k: int = 5,
    colpali_top_k_pages: int = 5,
    use_clip: bool = False,
) -> ComparisonResult:
    """
    Run both pipelines and compare results.

    Args:
        query: User query
        classic_retriever: LangChain retriever for classic RAG
        embeddings_dir: ColPali embeddings directory
        pages_dir: Page images directory
        client: Gemini client
        model: Model name
        classic_top_k: Top K for classic retrieval
        colpali_top_k_pages: Top K pages for ColPali

    Returns:
        ComparisonResult
    """
    # Run both pipelines
    classic_result = run_classic_pipeline(
        query=query,
        retriever=classic_retriever,
        client=client,
        model=model,
        top_k=classic_top_k,
    )

    colpali_result = run_colpali_pipeline(
        query=query,
        embeddings_dir=embeddings_dir,
        pages_dir=pages_dir,
        client=client,
        model=model,
        top_k_pages=colpali_top_k_pages,
        use_clip=use_clip,
    )

    # Compare
    return compare_results(classic_result, colpali_result)


def comparison_to_dict(comparison: ComparisonResult) -> Dict[str, Any]:
    """Convert comparison to JSON-serializable dict."""
    return {
        "query": comparison.query,
        "preferred_pipeline": comparison.preferred_pipeline,
        "reason": comparison.reason,
        "metrics": comparison.metrics,
        "classic": {
            "answer": comparison.classic.answer,
            "retrieval_time_ms": comparison.classic.retrieval_time_ms,
            "generation_time_ms": comparison.classic.generation_time_ms,
            "num_documents": len(comparison.classic.documents),
            "pages": [d.metadata.get("page") for d in comparison.classic.documents],
        },
        "colpali": {
            "answer": comparison.colpali.answer,
            "retrieval_time_ms": comparison.colpali.retrieval_time_ms,
            "generation_time_ms": comparison.colpali.generation_time_ms,
            "page_scores": comparison.colpali.page_scores,
            "total_patches_searched": comparison.colpali.total_patches_searched,
            "attribution_summary": comparison.colpali.attribution_summary,
        },
    }
