"""
Practice 6: ColPali-style End-to-End Multimodal RAG

Implements Docs/ Practice 6: Multimodal RAG with ColPali-like approach.

This module implements an alternative retrieval approach inspired by ColPali,
operating directly on visual document representations (page images and patches)
rather than extracted text.

Key differences from Classic RAG (Practices 1-5):
- No text extraction required
- Retrieval operates on visual patches
- Late interaction scoring (MaxSim-style)
- Visual context passed directly to VLM

This is an intern-level reference implementation for learning and demonstration.
"""

from rag.colpali.visual_ingestion import (
    PatchMetadata,
    PagePatches,
    extract_page_images,
    extract_patches,
    extract_all_patches,
    save_patches,
    load_patches,
    get_patch_image,
    reconstruct_page_region,
)

from rag.colpali.patch_embeddings import (
    PatchEmbedding,
    generate_patch_embeddings,
    generate_query_embedding,
    save_embeddings,
    load_embeddings,
    build_faiss_index,
    load_faiss_index,
)

from rag.colpali.late_interaction import (
    RetrievalResult,
    PatchScore,
    PageScore,
    late_interaction_search,
    late_interaction_search_faiss,
    aggregate_page_scores,
    retrieve_with_late_interaction,
    LateInteractionRetriever,
)

from rag.colpali.visual_generation import (
    VisualContext,
    VisualAnswer,
    collect_visual_context,
    generate_visual_answer,
    generate_visual_answer_stream,
    compare_with_text_answer,
)

from rag.colpali.source_attribution import (
    PatchAttribution,
    PageAttribution,
    AttributionResult,
    generate_attribution,
    save_attribution_images,
    attribution_to_dict,
)

from rag.colpali.comparison import (
    ClassicRAGResult,
    ColPaliResult,
    ComparisonResult,
    run_classic_pipeline,
    run_colpali_pipeline,
    run_comparison,
    comparison_to_dict,
)

__all__ = [
    # Visual ingestion
    "PatchMetadata",
    "PagePatches",
    "extract_page_images",
    "extract_patches",
    "extract_all_patches",
    "save_patches",
    "load_patches",
    "get_patch_image",
    "reconstruct_page_region",
    # Embeddings
    "PatchEmbedding",
    "generate_patch_embeddings",
    "generate_query_embedding",
    "save_embeddings",
    "load_embeddings",
    "build_faiss_index",
    "load_faiss_index",
    # Late interaction
    "RetrievalResult",
    "PatchScore",
    "PageScore",
    "late_interaction_search",
    "late_interaction_search_faiss",
    "aggregate_page_scores",
    "retrieve_with_late_interaction",
    "LateInteractionRetriever",
    # Generation
    "VisualContext",
    "VisualAnswer",
    "collect_visual_context",
    "generate_visual_answer",
    "generate_visual_answer_stream",
    "compare_with_text_answer",
    # Attribution
    "PatchAttribution",
    "PageAttribution",
    "AttributionResult",
    "generate_attribution",
    "save_attribution_images",
    "attribution_to_dict",
    # Comparison
    "ClassicRAGResult",
    "ColPaliResult",
    "ComparisonResult",
    "run_classic_pipeline",
    "run_colpali_pipeline",
    "run_comparison",
    "comparison_to_dict",
]
