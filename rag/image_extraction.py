"""
Image/Chart Extraction and Description for Multimodal RAG (Practice 5.2)

Extracts images (charts, graphs, figures) from PDFs and generates
textual descriptions using VLM for indexing and retrieval.

Features:
- Embedded image extraction
- Page rendering for full-page charts
- VLM-based description generation
- Chart type detection
- Key value extraction from visual data
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import base64


@dataclass
class ImageMetadata:
    """Metadata for an extracted image."""
    doc_id: str
    page: int
    figure_id: str
    caption: Optional[str] = None
    image_type: str = "unknown"  # chart, graph, figure, logo, photo
    chart_type: Optional[str] = None  # bar, line, pie, scatter, etc.
    extraction_method: str = "embedded"
    width: Optional[int] = None
    height: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    file_path: Optional[str] = None


@dataclass
class ExtractedImage:
    """Complete image representation with description for retrieval."""
    metadata: ImageMetadata
    # Textual description for retrieval
    description: str
    # Extracted key values/labels from the image
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    # VLM analysis result
    vlm_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "description": self.description,
            "extracted_data": self.extracted_data,
            "vlm_analysis": self.vlm_analysis,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedImage":
        return cls(
            metadata=ImageMetadata(**data["metadata"]),
            description=data["description"],
            extracted_data=data.get("extracted_data", {}),
            vlm_analysis=data.get("vlm_analysis"),
        )


def _detect_caption_near_image(
    page_text: str,
    bbox: Optional[Tuple],
) -> Optional[str]:
    """Detect figure caption from surrounding text."""
    patterns = [
        r"Figure\s+\d+[:\.]?\s*([^\n]+)",
        r"FIGURE\s+\d+[:\.]?\s*([^\n]+)",
        r"Fig\.\s*\d+[:\.]?\s*([^\n]+)",
        r"Chart\s+\d+[:\.]?\s*([^\n]+)",
        r"Graph\s+\d+[:\.]?\s*([^\n]+)",
        r"Exhibit\s+\d+[:\.]?\s*([^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, page_text)
        if match:
            caption = match.group(0).strip()
            if len(caption) > 300:
                caption = caption[:300] + "..."
            return caption

    return None


def _is_chart_or_graph(
    width: int,
    height: int,
    caption: Optional[str] = None,
    min_size: int = 80,
) -> bool:
    """Heuristic to determine if image is likely a chart/graph. Use min_size to include more figures."""
    if width < min_size or height < min_size:
        return False

    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio < 0.15 or aspect_ratio > 6:
        return False

    if caption:
        chart_keywords = ["chart", "graph", "figure", "trend", "growth", "comparison", "exhibit", "table"]
        if any(kw in caption.lower() for kw in chart_keywords):
            return True

    # Include any reasonably sized image (let VLM mark boilerplate later)
    if width >= min_size and height >= min_size:
        return True

    return False


def generate_image_description_vlm(
    image_bytes: bytes,
    image_ext: str,
    client,
    model: str,
    caption: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate detailed description of chart/graph using VLM.

    Returns dict with:
    - description: textual description for retrieval
    - chart_type: detected chart type
    - extracted_values: key values/labels extracted
    - axes: axis labels and units
    - trends: detected trends or insights
    """
    from google.genai import types

    prompt = """Analyze this image from a financial report. Provide a detailed description for retrieval.

First, decide if this is SUBSTANTIVE content (charts, data, maps, org charts, climate/sustainability figures) or BOILERPLATE (contact info, logo, footer, legal disclaimer, empty decoration).

If this is a chart or graph:
1. Identify the chart type (bar, line, pie, area, scatter, etc.)
2. Extract axis labels and units
3. List key data points and labels
4. Describe the main trend or insight
5. Mention topics covered: e.g. climate, sustainability, investments, regional distribution, financial metrics, governance

If this is boilerplate (contact block, logo, footer):
1. Set content_type to "boilerplate"
2. Describe briefly: e.g. "Auditor contact details", "Company logo", "Legal disclaimer"
3. Do NOT use words like "investments", "climate", "sustainability" unless the image actually shows that

Return a JSON object with:
- description: Text description (2-4 sentences). For boilerplate: one short sentence only.
- chart_type: Chart type or "figure" or "boilerplate"
- image_type: "chart", "graph", "diagram", "figure", "table_image", "contact", "logo", or "other"
- content_type: "substantive" or "boilerplate"
- topics: List of main topics if substantive (e.g. ["climate", "investments"], ["financial results"], [] for boilerplate)
- extracted_values: Object mapping labels to values where visible
- axes: Object with "x" and "y" axis labels/units if applicable
- trends: Brief description of trends (only for charts)
- confidence: "high", "medium", or "low"
"""

    if caption:
        prompt += f"\n\nContext - detected caption: {caption}"

    response_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "chart_type": {"type": "string"},
            "image_type": {"type": "string"},
            "content_type": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "extracted_values": {"type": "object"},
            "axes": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                },
            },
            "trends": {"type": "string"},
            "confidence": {"type": "string"},
        },
        "required": ["description", "chart_type", "image_type"],
    }

    mime_type = f"image/{image_ext}" if image_ext != "jpg" else "image/jpeg"
    parts = [types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]

    try:
        from rag.gemini_client import generate_json
        result = generate_json(
            client=client,
            model=model,
            prompt=prompt,
            response_schema=response_schema,
            parts=parts,
        )
        return result
    except Exception as e:
        return {
            "description": caption or "Image from financial report",
            "chart_type": "unknown",
            "image_type": "unknown",
            "content_type": "substantive",
            "topics": [],
            "extracted_values": {},
            "error": str(e),
        }


def _page_has_figure_caption(page_text: str) -> bool:
    """True if page text contains a figure/chart caption (vector-drawn figures)."""
    if not page_text:
        return False
    patterns = [
        r"Figure\s+\d+",
        r"FIGURE\s+\d+",
        r"Fig\.\s*\d+",
        r"Chart\s+\d+",
        r"Graph\s+\d+",
        r"Exhibit\s+\d+",
    ]
    return any(re.search(p, page_text, re.IGNORECASE) for p in patterns)


def extract_images_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    doc_id: str = "doc",
    client=None,
    model: str = "gemini-2.0-flash",
    use_vlm: bool = True,
    min_width: int = 80,
    min_height: int = 80,
    extract_page_renders: bool = True,
    page_render_dpi: int = 150,
) -> List[ExtractedImage]:
    """
    Extract images from PDF (embedded raster + vector via page render) and generate descriptions.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images
        doc_id: Document identifier
        client: Gemini client for VLM analysis
        model: Model to use for VLM
        use_vlm: Whether to use VLM for description generation
        min_width: Minimum image width for embedded images
        min_height: Minimum image height for embedded images
        extract_page_renders: If True, render pages with figure captions to capture vector graphics
        page_render_dpi: DPI for page render (vector → raster)

    Returns:
        List of ExtractedImage objects
    """
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    images: List[ExtractedImage] = []
    global_fig_idx = 0
    pages_with_embedded: set[int] = set()

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        page_text = page.get_text("text")

        # Extract embedded images
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]

            try:
                extracted = doc.extract_image(xref)
            except Exception:
                continue

            ext = extracted.get("ext", "png")
            width = extracted.get("width", 0)
            height = extracted.get("height", 0)
            image_bytes = extracted.get("image")

            if not image_bytes:
                continue

            if width < min_width or height < min_height:
                continue

            caption = _detect_caption_near_image(page_text, None)
            if not _is_chart_or_graph(width, height, caption, min_size=min(min_width, min_height)):
                continue

            global_fig_idx += 1
            figure_id = f"figure_{global_fig_idx:03d}"

            # Save image file
            filename = f"{figure_id}.{ext}"
            image_path = output_dir / filename
            image_path.write_bytes(image_bytes)

            # Generate description
            if use_vlm and client:
                vlm_result = generate_image_description_vlm(
                    image_bytes=image_bytes,
                    image_ext=ext,
                    client=client,
                    model=model,
                    caption=caption,
                )
            else:
                vlm_result = {
                    "description": caption or f"Image from page {page_idx + 1}",
                    "chart_type": "unknown",
                    "image_type": "unknown",
                }

            # Build description for retrieval (boilerplate clearly marked so semantic search won't match topic queries)
            content_type = (vlm_result.get("content_type") or "substantive").lower()
            description_parts = [
                f"[FIGURE] Page {page_idx + 1}, {figure_id}",
            ]
            if content_type == "boilerplate":
                description_parts.append("[BOILERPLATE: contact/logo/footer - not substantive content.]")
            if caption:
                description_parts.append(f"Caption: {caption}")
            description_parts.append(vlm_result.get("description", ""))
            if vlm_result.get("topics"):
                description_parts.append("Topics: " + ", ".join(vlm_result["topics"]))

            if vlm_result.get("chart_type") and vlm_result["chart_type"] != "unknown":
                description_parts.append(f"Chart type: {vlm_result['chart_type']}")

            if vlm_result.get("axes"):
                axes = vlm_result["axes"]
                if axes.get("x"):
                    description_parts.append(f"X-axis: {axes['x']}")
                if axes.get("y"):
                    description_parts.append(f"Y-axis: {axes['y']}")

            if vlm_result.get("trends"):
                description_parts.append(f"Trend: {vlm_result['trends']}")

            # Create metadata
            metadata = ImageMetadata(
                doc_id=doc_id,
                page=page_idx + 1,
                figure_id=figure_id,
                caption=caption,
                image_type=vlm_result.get("image_type", "unknown"),
                chart_type=vlm_result.get("chart_type"),
                extraction_method="embedded",
                width=width,
                height=height,
                file_path=str(image_path),
            )

            images.append(ExtractedImage(
                metadata=metadata,
                description="\n".join(description_parts),
                extracted_data=vlm_result.get("extracted_values", {}),
                vlm_analysis=vlm_result,
            ))
            pages_with_embedded.add(page_idx + 1)

    # Vector graphics: render pages that have figure/chart captions but no embedded image from that page
    if extract_page_renders and client and use_vlm:
        for page_idx in range(doc.page_count):
            page_num = page_idx + 1
            if page_num in pages_with_embedded:
                continue
            page = doc.load_page(page_idx)
            page_text = page.get_text("text")
            if not _page_has_figure_caption(page_text):
                continue

            mat = fitz.Matrix(page_render_dpi / 72, page_render_dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image_bytes = pix.tobytes("png")
            width, height = pix.width, pix.height

            global_fig_idx += 1
            figure_id = f"figure_{global_fig_idx:03d}"
            caption = _detect_caption_near_image(page_text, None)
            filename = f"{figure_id}_page{page_num}.png"
            image_path = output_dir / filename
            image_path.write_bytes(image_bytes)

            vlm_result = generate_image_description_vlm(
                image_bytes=image_bytes,
                image_ext="png",
                client=client,
                model=model,
                caption=caption,
            )
            content_type = (vlm_result.get("content_type") or "substantive").lower()
            description_parts = [
                f"[FIGURE] Page {page_num}, {figure_id} (page render, vector content)",
            ]
            if content_type == "boilerplate":
                description_parts.append("[BOILERPLATE: contact/logo/footer - not substantive content.]")
            if caption:
                description_parts.append(f"Caption: {caption}")
            description_parts.append(vlm_result.get("description", ""))
            if vlm_result.get("topics"):
                description_parts.append("Topics: " + ", ".join(vlm_result["topics"]))
            if vlm_result.get("chart_type") and vlm_result["chart_type"] != "unknown":
                description_parts.append(f"Chart type: {vlm_result['chart_type']}")
            if vlm_result.get("axes"):
                axes = vlm_result["axes"]
                if axes.get("x"):
                    description_parts.append(f"X-axis: {axes['x']}")
                if axes.get("y"):
                    description_parts.append(f"Y-axis: {axes['y']}")
            if vlm_result.get("trends"):
                description_parts.append(f"Trend: {vlm_result['trends']}")

            metadata = ImageMetadata(
                doc_id=doc_id,
                page=page_num,
                figure_id=figure_id,
                caption=caption,
                image_type=vlm_result.get("image_type", "unknown"),
                chart_type=vlm_result.get("chart_type"),
                extraction_method="page_render",
                width=width,
                height=height,
                file_path=str(image_path),
            )
            images.append(ExtractedImage(
                metadata=metadata,
                description="\n".join(description_parts),
                extracted_data=vlm_result.get("extracted_values", {}),
                vlm_analysis=vlm_result,
            ))

    doc.close()
    return images


def render_page_as_image(
    pdf_path: Path,
    page_num: int,
    output_path: Path,
    dpi: int = 150,
) -> bytes:
    """Render a PDF page as an image."""
    import fitz

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)  # 0-indexed

    # Render at specified DPI
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    # Save as PNG
    image_bytes = pix.tobytes("png")
    output_path.write_bytes(image_bytes)

    doc.close()
    return image_bytes


def save_images(images: List[ExtractedImage], output_dir: Path) -> None:
    """Save extracted image metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata index
    metadata_rows = []
    for img in images:
        # Save individual image JSON
        img_file = output_dir / f"{img.metadata.figure_id}.json"
        img_file.write_text(
            json.dumps(img.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metadata_rows.append({
            "figure_id": img.metadata.figure_id,
            "page": img.metadata.page,
            "caption": img.metadata.caption,
            "image_type": img.metadata.image_type,
            "chart_type": img.metadata.chart_type,
            "file_path": img.metadata.file_path,
        })

    metadata_file = output_dir / "images_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_images(images_dir: Path) -> List[ExtractedImage]:
    """Load extracted images from disk."""
    images = []
    metadata_file = images_dir / "images_metadata.json"

    if not metadata_file.exists():
        return images

    with metadata_file.open("r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    for meta in metadata_list:
        img_file = images_dir / f"{meta['figure_id']}.json"
        if img_file.exists():
            with img_file.open("r", encoding="utf-8") as f:
                img_data = json.load(f)
                images.append(ExtractedImage.from_dict(img_data))

    return images
