"""
Source Attribution with Patch Visualization for ColPali-style RAG (Phase 6)

Provides visual evidence for retrieval results by:
1. Highlighting relevant patches on page images
2. Creating heat maps of patch relevance
3. Generating attribution summaries

This enables users to see exactly which visual regions
contributed to the retrieved results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io

from rag.colpali.late_interaction import RetrievalResult, PageScore, PatchScore
from rag.colpali.patch_embeddings import PatchEmbedding


@dataclass
class PatchAttribution:
    """Attribution for a single patch."""
    patch_id: str
    page: int
    score: float
    rank: int
    # Bounding box (normalized 0-1)
    x: float
    y: float
    width: float
    height: float
    # Visual info
    highlight_color: Tuple[int, int, int, int]


@dataclass
class PageAttribution:
    """Attribution for a page showing relevant regions."""
    page: int
    doc_id: str
    page_score: float
    patches: List[PatchAttribution]
    # Generated images
    original_image_path: Optional[str] = None
    highlighted_image_bytes: Optional[bytes] = None
    heatmap_image_bytes: Optional[bytes] = None


@dataclass
class AttributionResult:
    """Full attribution result with visualizations."""
    query: str
    page_attributions: List[PageAttribution]
    total_relevant_patches: int
    summary: str


def score_to_color(score: float, alpha: int = 150) -> Tuple[int, int, int, int]:
    """
    Convert relevance score to RGBA color.

    High scores = green, medium = yellow, low = red
    """
    if score > 0.7:
        return (0, 255, 0, alpha)  # Green
    elif score > 0.5:
        return (255, 255, 0, alpha)  # Yellow
    elif score > 0.3:
        return (255, 165, 0, alpha)  # Orange
    else:
        return (255, 0, 0, alpha)  # Red


def create_highlighted_page(
    page_image_path: str,
    patches: List[PatchAttribution],
    show_scores: bool = True,
) -> Optional[bytes]:
    """
    Create page image with highlighted patches.

    Args:
        page_image_path: Path to original page image
        patches: List of patches to highlight
        show_scores: Whether to show score labels

    Returns:
        PNG image bytes or None if failed
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    try:
        # Load image
        img = Image.open(page_image_path).convert("RGBA")
        width, height = img.size

        # Create overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw patch highlights
        for patch in patches:
            # Convert normalized coords to pixels
            x1 = int(patch.x * width)
            y1 = int(patch.y * height)
            x2 = int((patch.x + patch.width) * width)
            y2 = int((patch.y + patch.height) * height)

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], fill=patch.highlight_color)

            # Draw border
            border_color = tuple(patch.highlight_color[:3]) + (255,)
            draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)

            # Draw score label
            if show_scores:
                label = f"{patch.score:.2f}"
                try:
                    font = ImageFont.load_default()
                    draw.text((x1 + 5, y1 + 5), label, fill=(0, 0, 0, 255), font=font)
                except Exception:
                    pass

        # Composite
        result = Image.alpha_composite(img, overlay)

        # Convert to bytes
        output = io.BytesIO()
        result.convert("RGB").save(output, format="PNG")
        return output.getvalue()

    except Exception:
        return None


def create_heatmap(
    page_image_path: str,
    patches: List[PatchAttribution],
    grid_size: Tuple[int, int] = (16, 12),
) -> Optional[bytes]:
    """
    Create heat map overlay showing patch relevance.

    Args:
        page_image_path: Path to original page image
        patches: List of patches with scores
        grid_size: Grid dimensions for heat map

    Returns:
        PNG image bytes or None if failed
    """
    try:
        from PIL import Image, ImageDraw
        import numpy as np
    except ImportError:
        return None

    try:
        # Load image
        img = Image.open(page_image_path).convert("RGBA")
        width, height = img.size

        # Create heat map grid
        heat_grid = np.zeros(grid_size, dtype=np.float32)

        for patch in patches:
            # Map patch to grid cell
            grid_x = int(patch.x * grid_size[0])
            grid_y = int(patch.y * grid_size[1])

            grid_x = min(grid_x, grid_size[0] - 1)
            grid_y = min(grid_y, grid_size[1] - 1)

            # Add score to cell (max aggregation)
            heat_grid[grid_y, grid_x] = max(heat_grid[grid_y, grid_x], patch.score)

        # Normalize heat map
        if heat_grid.max() > 0:
            heat_grid = heat_grid / heat_grid.max()

        # Create overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cell_width = width / grid_size[0]
        cell_height = height / grid_size[1]

        for gy in range(grid_size[1]):
            for gx in range(grid_size[0]):
                value = heat_grid[gy, gx]
                if value > 0.1:  # Only draw visible cells
                    # Color based on value
                    r = int(255 * value)
                    g = int(255 * (1 - value))
                    b = 0
                    a = int(100 * value)

                    x1 = int(gx * cell_width)
                    y1 = int(gy * cell_height)
                    x2 = int((gx + 1) * cell_width)
                    y2 = int((gy + 1) * cell_height)

                    draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, a))

        # Composite
        result = Image.alpha_composite(img, overlay)

        # Convert to bytes
        output = io.BytesIO()
        result.convert("RGB").save(output, format="PNG")
        return output.getvalue()

    except Exception:
        return None


def generate_attribution(
    retrieval_result: RetrievalResult,
    pages_dir: Path,
    max_pages: int = 5,
    max_patches_per_page: int = 10,
    create_visualizations: bool = True,
) -> AttributionResult:
    """
    Generate full attribution result with visualizations.

    Args:
        retrieval_result: Result from late interaction retrieval
        pages_dir: Directory containing page images
        max_pages: Maximum pages to include
        max_patches_per_page: Maximum patches per page
        create_visualizations: Whether to create highlighted images

    Returns:
        AttributionResult with visualizations
    """
    page_attributions = []
    total_patches = 0

    for page_score in retrieval_result.page_scores[:max_pages]:
        # Find page image
        page_image = pages_dir / f"{page_score.doc_id}_page_{page_score.page}.png"
        if not page_image.exists():
            page_image = pages_dir / f"page_{page_score.page}.png"

        # Build patch attributions
        patch_attrs = []
        for rank, ps in enumerate(page_score.top_patches[:max_patches_per_page], 1):
            patch_attrs.append(PatchAttribution(
                patch_id=ps.patch_id,
                page=ps.page,
                score=ps.score,
                rank=rank,
                x=ps.x,
                y=ps.y,
                width=ps.width,
                height=ps.height,
                highlight_color=score_to_color(ps.score),
            ))
            total_patches += 1

        # Create visualizations
        highlighted_bytes = None
        heatmap_bytes = None

        if create_visualizations and page_image.exists():
            highlighted_bytes = create_highlighted_page(str(page_image), patch_attrs)
            heatmap_bytes = create_heatmap(str(page_image), patch_attrs)

        page_attributions.append(PageAttribution(
            page=page_score.page,
            doc_id=page_score.doc_id,
            page_score=page_score.score,
            patches=patch_attrs,
            original_image_path=str(page_image) if page_image.exists() else None,
            highlighted_image_bytes=highlighted_bytes,
            heatmap_image_bytes=heatmap_bytes,
        ))

    # Generate summary
    summary = generate_attribution_summary(retrieval_result, page_attributions)

    return AttributionResult(
        query=retrieval_result.query,
        page_attributions=page_attributions,
        total_relevant_patches=total_patches,
        summary=summary,
    )


def generate_attribution_summary(
    retrieval_result: RetrievalResult,
    page_attributions: List[PageAttribution],
) -> str:
    """Generate text summary of attribution."""
    if not page_attributions:
        return "No relevant pages found."

    lines = [f"Query: \"{retrieval_result.query}\"", ""]

    # Overview
    lines.append(f"Found {len(page_attributions)} relevant page(s) from {retrieval_result.total_patches_searched} patches.")
    lines.append("")

    # Per-page summary
    for pa in page_attributions:
        lines.append(f"**Page {pa.page}** (score: {pa.page_score:.3f})")
        lines.append(f"  - {len(pa.patches)} relevant region(s)")

        if pa.patches:
            top = pa.patches[0]
            lines.append(f"  - Top region: ({top.x:.2f}, {top.y:.2f}), score: {top.score:.3f}")

        lines.append("")

    return "\n".join(lines)


def save_attribution_images(
    attribution: AttributionResult,
    output_dir: Path,
    prefix: str = "",
) -> Dict[str, Path]:
    """
    Save attribution images to disk.

    Args:
        attribution: Attribution result
        output_dir: Directory to save images
        prefix: Filename prefix

    Returns:
        Dict mapping page numbers to saved paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    for pa in attribution.page_attributions:
        # Save highlighted version
        if pa.highlighted_image_bytes:
            path = output_dir / f"{prefix}page_{pa.page}_highlighted.png"
            path.write_bytes(pa.highlighted_image_bytes)
            saved[f"page_{pa.page}_highlighted"] = path

        # Save heatmap version
        if pa.heatmap_image_bytes:
            path = output_dir / f"{prefix}page_{pa.page}_heatmap.png"
            path.write_bytes(pa.heatmap_image_bytes)
            saved[f"page_{pa.page}_heatmap"] = path

    return saved


def attribution_to_dict(attribution: AttributionResult) -> Dict[str, Any]:
    """Convert attribution to JSON-serializable dict."""
    return {
        "query": attribution.query,
        "total_relevant_patches": attribution.total_relevant_patches,
        "summary": attribution.summary,
        "pages": [
            {
                "page": pa.page,
                "doc_id": pa.doc_id,
                "score": pa.page_score,
                "num_patches": len(pa.patches),
                "has_highlighted_image": pa.highlighted_image_bytes is not None,
                "has_heatmap_image": pa.heatmap_image_bytes is not None,
                "patches": [
                    {
                        "patch_id": p.patch_id,
                        "rank": p.rank,
                        "score": p.score,
                        "x": p.x,
                        "y": p.y,
                        "width": p.width,
                        "height": p.height,
                    }
                    for p in pa.patches
                ],
            }
            for pa in attribution.page_attributions
        ],
    }
