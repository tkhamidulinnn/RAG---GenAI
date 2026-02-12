"""
Visual Context Generation for ColPali-style RAG (Phase 6)

Generates answers using VLM (Vision Language Model) by:
1. Taking top retrieved patches/pages
2. Sending visual context to VLM
3. Grounding answers in visual evidence

This is the "generation" phase of ColPali-style RAG where
visual understanding directly informs the answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

from rag.colpali.late_interaction import RetrievalResult, PageScore, PatchScore


@dataclass
class VisualContext:
    """Visual context for generation."""
    page: int
    doc_id: str
    image_path: str
    relevant_patches: List[PatchScore]
    confidence: float


@dataclass
class VisualAnswer:
    """Answer generated from visual context."""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    reasoning: Optional[str] = None


def collect_visual_context(
    retrieval_result: RetrievalResult,
    pages_dir: Path,
    max_pages: int = 3,
) -> List[VisualContext]:
    """
    Collect visual context from retrieval results.

    Args:
        retrieval_result: Result from late interaction retrieval
        pages_dir: Directory containing page images
        max_pages: Maximum number of pages to include

    Returns:
        List of VisualContext objects
    """
    contexts = []

    for page_score in retrieval_result.page_scores[:max_pages]:
        # Find page image (visual_ingestion saves as page_001.png, page_002.png, ...)
        page_image = pages_dir / f"{page_score.doc_id}_page_{page_score.page}.png"
        if not page_image.exists():
            page_image = pages_dir / f"page_{page_score.page:03d}.png"
        if not page_image.exists():
            page_image = pages_dir / f"page_{page_score.page}.png"

        if not page_image.exists():
            continue

        contexts.append(VisualContext(
            page=page_score.page,
            doc_id=page_score.doc_id,
            image_path=str(page_image),
            relevant_patches=page_score.top_patches,
            confidence=page_score.score,
        ))

    return contexts


def build_visual_qa_prompt(query: str, contexts: List[VisualContext]) -> str:
    """
    Build prompt for visual QA.

    Args:
        query: User query
        contexts: Visual contexts to include

    Returns:
        Formatted prompt
    """
    prompt = f"""You are a document analysis assistant using visual understanding.

QUERY: {query}

You have been provided with {len(contexts)} page image(s) from a document.
The most relevant regions within each page have been identified.

INSTRUCTIONS:
1. Examine the provided page images carefully
2. Focus on the highlighted/relevant regions
3. Extract information that answers the query
4. Cite specific evidence (page number, visual location)

RESPONSE FORMAT:
- Start with a direct answer to the query
- Support with specific visual evidence
- If information is not found, say "NOT FOUND IN VISUAL CONTEXT"
- Be precise - cite exact text/numbers visible in the images

RELEVANT PAGES:
"""

    for i, ctx in enumerate(contexts, 1):
        prompt += f"\nPage {ctx.page} (confidence: {ctx.confidence:.2f}):"
        if ctx.relevant_patches:
            prompt += f"\n  - {len(ctx.relevant_patches)} relevant region(s) identified"
            for patch in ctx.relevant_patches[:3]:
                prompt += f"\n  - Region at ({patch.x:.2f}, {patch.y:.2f}), score: {patch.score:.2f}"

    prompt += "\n\nPlease analyze the images and answer the query."

    return prompt


def generate_visual_answer(
    client,
    model: str,
    query: str,
    contexts: List[VisualContext],
    stream: bool = False,
) -> VisualAnswer | Iterator[str]:
    """
    Generate answer using VLM with visual context.

    Args:
        client: Gemini client
        model: Model name (e.g., "gemini-2.0-flash")
        query: User query
        contexts: Visual contexts with page images
        stream: Whether to stream the response

    Returns:
        VisualAnswer or stream of tokens
    """
    from google.genai import types

    if not contexts:
        if stream:
            yield "No visual context available. Please ensure page images are indexed."
            return
        return VisualAnswer(
            answer="No visual context available.",
            confidence=0.0,
            sources=[],
        )

    # Build prompt
    prompt = build_visual_qa_prompt(query, contexts)

    # Build parts with images (google.genai Part uses keyword args: text=, data=, mime_type=)
    parts = [types.Part.from_text(text=prompt)]

    for ctx in contexts:
        try:
            image_bytes = Path(ctx.image_path).read_bytes()
            parts.append(types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            ))
            parts.append(types.Part.from_text(text=f"[Above: Page {ctx.page}]"))
        except Exception as e:
            parts.append(types.Part.from_text(text=f"[Page {ctx.page} image unavailable: {e}]"))

    # Generate response
    try:
        if stream:
            response = client.models.generate_content_stream(
                model=model,
                contents=parts,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        else:
            response = client.models.generate_content(
                model=model,
                contents=parts,
            )

            answer_text = response.text if response.text else "No answer generated."

            # Build sources
            sources = [
                {
                    "page": ctx.page,
                    "doc_id": ctx.doc_id,
                    "confidence": ctx.confidence,
                    "num_patches": len(ctx.relevant_patches),
                }
                for ctx in contexts
            ]

            # Estimate confidence from context scores
            avg_confidence = sum(ctx.confidence for ctx in contexts) / len(contexts)

            return VisualAnswer(
                answer=answer_text,
                confidence=avg_confidence,
                sources=sources,
            )

    except Exception as e:
        if stream:
            yield f"Error generating answer: {e}"
            return
        return VisualAnswer(
            answer=f"Error generating answer: {e}",
            confidence=0.0,
            sources=[],
        )


def generate_visual_answer_stream(
    client,
    model: str,
    query: str,
    contexts: List[VisualContext],
) -> Iterator[str]:
    """
    Stream answer generation with visual context.

    Args:
        client: Gemini client
        model: Model name
        query: User query
        contexts: Visual contexts

    Yields:
        Answer tokens
    """
    yield from generate_visual_answer(client, model, query, contexts, stream=True)


def generate_with_highlighted_regions(
    client,
    model: str,
    query: str,
    retrieval_result: RetrievalResult,
    pages_dir: Path,
    patches_dir: Path,
    highlight_patches: bool = True,
) -> VisualAnswer:
    """
    Generate answer with highlighted relevant regions.

    This creates annotated page images showing which patches
    contributed to the retrieval, then sends to VLM.

    Args:
        client: Gemini client
        model: Model name
        query: User query
        retrieval_result: Retrieval results with patch scores
        pages_dir: Directory with page images
        patches_dir: Directory with patch images
        highlight_patches: Whether to highlight patches on page

    Returns:
        VisualAnswer with region attribution
    """
    from rag.colpali.visual_ingestion import load_page_patches, reconstruct_page_region

    # Collect contexts
    contexts = collect_visual_context(retrieval_result, pages_dir)

    if not contexts:
        return VisualAnswer(
            answer="No visual context available.",
            confidence=0.0,
            sources=[],
        )

    # If highlighting, create annotated images
    if highlight_patches:
        annotated_contexts = []

        for ctx in contexts:
            # Load page patches
            patches_file = patches_dir / f"{ctx.doc_id}_page_{ctx.page}_patches.json"
            if patches_file.exists():
                try:
                    page_patches = load_page_patches(patches_file)

                    # Get patch IDs to highlight
                    highlight_ids = [p.patch_id for p in ctx.relevant_patches]

                    # Reconstruct with highlights
                    highlighted_bytes = reconstruct_page_region(
                        page_patches,
                        highlight_ids,
                        highlight_color=(255, 255, 0, 100),  # Yellow with alpha
                    )

                    if highlighted_bytes:
                        # Save temporary highlighted image
                        temp_path = patches_dir / f"temp_highlighted_{ctx.page}.png"
                        temp_path.write_bytes(highlighted_bytes)

                        # Update context with highlighted image
                        ctx = VisualContext(
                            page=ctx.page,
                            doc_id=ctx.doc_id,
                            image_path=str(temp_path),
                            relevant_patches=ctx.relevant_patches,
                            confidence=ctx.confidence,
                        )
                except Exception:
                    pass  # Use original image

            annotated_contexts.append(ctx)

        contexts = annotated_contexts

    # Generate answer
    return generate_visual_answer(client, model, query, contexts, stream=False)


def compare_with_text_answer(
    visual_answer: VisualAnswer,
    text_answer: str,
) -> Dict[str, Any]:
    """
    Compare visual-based answer with text-based answer.

    Useful for evaluating ColPali vs classic RAG.

    Args:
        visual_answer: Answer from visual pipeline
        text_answer: Answer from classic text pipeline

    Returns:
        Comparison dictionary
    """
    # Simple comparison metrics
    visual_len = len(visual_answer.answer)
    text_len = len(text_answer)

    # Check for key differences
    visual_words = set(visual_answer.answer.lower().split())
    text_words = set(text_answer.lower().split())

    common_words = visual_words & text_words
    visual_only = visual_words - text_words
    text_only = text_words - visual_words

    return {
        "visual_answer_length": visual_len,
        "text_answer_length": text_len,
        "common_words": len(common_words),
        "visual_unique_words": len(visual_only),
        "text_unique_words": len(text_only),
        "visual_confidence": visual_answer.confidence,
        "visual_sources": len(visual_answer.sources),
        "overlap_ratio": len(common_words) / max(len(visual_words), len(text_words), 1),
    }
