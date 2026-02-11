"""
Visual Document Ingestion for ColPali-style RAG (Phase 6)

Converts PDF pages into images and segments them into patches
for visual retrieval. This approach bypasses text extraction
and operates directly on visual representations.

Key concepts:
- Page images: Full rendered PDF pages as images
- Patches: Grid-based segments of page images
- Patch metadata: Location, size, and document info for each patch
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import io


@dataclass
class PatchMetadata:
    """Metadata for a single image patch."""
    doc_id: str
    page: int
    patch_id: str
    # Coordinates relative to page (normalized 0-1)
    x: float
    y: float
    width: float
    height: float
    # Grid position
    row: int
    col: int
    # Absolute pixel coordinates
    pixel_x: int
    pixel_y: int
    pixel_width: int
    pixel_height: int
    # File path to patch image
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatchMetadata":
        return cls(**data)


@dataclass
class PagePatches:
    """Collection of patches from a single page."""
    doc_id: str
    page: int
    page_width: int
    page_height: int
    grid_rows: int
    grid_cols: int
    patches: List[PatchMetadata] = field(default_factory=list)
    page_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page": self.page,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "patches": [p.to_dict() for p in self.patches],
            "page_image_path": self.page_image_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PagePatches":
        patches = [PatchMetadata.from_dict(p) for p in data.get("patches", [])]
        return cls(
            doc_id=data["doc_id"],
            page=data["page"],
            page_width=data["page_width"],
            page_height=data["page_height"],
            grid_rows=data["grid_rows"],
            grid_cols=data["grid_cols"],
            patches=patches,
            page_image_path=data.get("page_image_path"),
        )


def extract_page_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 150,
    doc_id: str = "doc",
) -> List[Dict[str, Any]]:
    """
    Render all PDF pages as images.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save page images
        dpi: Resolution for rendering (higher = better quality, larger files)
        doc_id: Document identifier

    Returns:
        List of dicts with page info and image paths
    """
    import fitz  # PyMuPDF

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_info = []

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)

        # Render page at specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # Save as PNG
        page_num = page_idx + 1
        filename = f"page_{page_num:03d}.png"
        image_path = output_dir / filename
        pix.save(str(image_path))

        page_info.append({
            "doc_id": doc_id,
            "page": page_num,
            "width": pix.width,
            "height": pix.height,
            "dpi": dpi,
            "image_path": str(image_path),
        })

    doc.close()
    return page_info


def extract_patches(
    page_image_path: Path,
    page_num: int,
    doc_id: str = "doc",
    grid_rows: int = 2,
    grid_cols: int = 2,
    output_dir: Optional[Path] = None,
    overlap: float = 0.1,
) -> PagePatches:
    """
    Extract grid-based patches from a page image.

    The page is divided into a grid of (grid_rows x grid_cols) patches.
    Optional overlap between patches improves retrieval at boundaries.

    Args:
        page_image_path: Path to the page image
        page_num: Page number
        doc_id: Document identifier
        grid_rows: Number of rows in the patch grid
        grid_cols: Number of columns in the patch grid
        output_dir: Directory to save patch images (None = don't save)
        overlap: Fraction of overlap between adjacent patches (0-0.5)

    Returns:
        PagePatches object containing all patch metadata
    """
    from PIL import Image

    # Load page image
    img = Image.open(page_image_path)
    page_width, page_height = img.size

    # Calculate base patch size
    base_patch_width = page_width // grid_cols
    base_patch_height = page_height // grid_rows

    # Calculate overlap in pixels
    overlap_x = int(base_patch_width * overlap)
    overlap_y = int(base_patch_height * overlap)

    # Actual patch size with overlap
    patch_width = base_patch_width + overlap_x
    patch_height = base_patch_height + overlap_y

    patches = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate pixel coordinates
            pixel_x = col * base_patch_width
            pixel_y = row * base_patch_height

            # Adjust for overlap (don't exceed image bounds)
            x_end = min(pixel_x + patch_width, page_width)
            y_end = min(pixel_y + patch_height, page_height)

            # Create patch ID
            patch_id = f"p{page_num:03d}_r{row}_c{col}"

            # Extract patch image
            patch_img = img.crop((pixel_x, pixel_y, x_end, y_end))

            # Save patch if output_dir provided
            file_path = None
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                patch_filename = f"{patch_id}.png"
                file_path = str(output_dir / patch_filename)
                patch_img.save(file_path)

            # Create metadata with normalized coordinates
            metadata = PatchMetadata(
                doc_id=doc_id,
                page=page_num,
                patch_id=patch_id,
                x=pixel_x / page_width,
                y=pixel_y / page_height,
                width=(x_end - pixel_x) / page_width,
                height=(y_end - pixel_y) / page_height,
                row=row,
                col=col,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                pixel_width=x_end - pixel_x,
                pixel_height=y_end - pixel_y,
                file_path=file_path,
            )
            patches.append(metadata)

    return PagePatches(
        doc_id=doc_id,
        page=page_num,
        page_width=page_width,
        page_height=page_height,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        patches=patches,
        page_image_path=str(page_image_path),
    )


def extract_all_patches(
    pdf_path: Path,
    output_dir: Path,
    doc_id: str = "doc",
    dpi: int = 150,
    grid_rows: int = 2,
    grid_cols: int = 2,
    overlap: float = 0.1,
    save_patches_images: bool = True,
) -> List[PagePatches]:
    """
    Full visual ingestion pipeline: PDF -> page images -> patches.

    Args:
        pdf_path: Path to PDF file
        output_dir: Base output directory
        doc_id: Document identifier
        dpi: Resolution for page rendering
        grid_rows: Number of rows in patch grid
        grid_cols: Number of columns in patch grid
        overlap: Overlap fraction between patches
        save_patches_images: Whether to save individual patch images

    Returns:
        List of PagePatches for all pages
    """
    pages_dir = output_dir / "pages"
    patches_dir = output_dir / "patches" if save_patches_images else None

    # Step 1: Render all pages as images
    print(f"  Rendering PDF pages at {dpi} DPI...")
    page_infos = extract_page_images(pdf_path, pages_dir, dpi=dpi, doc_id=doc_id)
    print(f"  Rendered {len(page_infos)} pages")

    # Step 2: Extract patches from each page
    all_patches = []
    total_patch_count = 0

    print(f"  Extracting patches (grid: {grid_rows}x{grid_cols}, overlap: {overlap:.0%})...")
    for page_info in page_infos:
        page_patches = extract_patches(
            page_image_path=Path(page_info["image_path"]),
            page_num=page_info["page"],
            doc_id=doc_id,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            output_dir=patches_dir,
            overlap=overlap,
        )
        all_patches.append(page_patches)
        total_patch_count += len(page_patches.patches)

    print(f"  Extracted {total_patch_count} patches from {len(page_infos)} pages")
    return all_patches


def save_patches(patches_list: List[PagePatches], output_dir: Path) -> None:
    """Save patch metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full metadata
    metadata_file = output_dir / "patches_metadata.json"
    data = {
        "pages": [p.to_dict() for p in patches_list],
        "total_patches": sum(len(p.patches) for p in patches_list),
        "total_pages": len(patches_list),
    }
    metadata_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_patches(patches_dir: Path) -> List[PagePatches]:
    """Load patch metadata from disk."""
    metadata_file = patches_dir / "patches_metadata.json"

    if not metadata_file.exists():
        return []

    with metadata_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [PagePatches.from_dict(p) for p in data.get("pages", [])]


def get_patch_image(patch: PatchMetadata) -> Optional[bytes]:
    """Load a patch image as bytes."""
    if not patch.file_path:
        return None

    path = Path(patch.file_path)
    if not path.exists():
        return None

    return path.read_bytes()


def reconstruct_page_region(
    page_patches: PagePatches,
    patches_to_highlight: List[PatchMetadata],
    highlight_color: Tuple[int, int, int] = (255, 0, 0),
    highlight_alpha: float = 0.3,
) -> bytes:
    """
    Reconstruct a page image with highlighted patches.

    Args:
        page_patches: PagePatches containing page info
        patches_to_highlight: List of patches to highlight
        highlight_color: RGB color for highlighting
        highlight_alpha: Transparency of highlight overlay

    Returns:
        PNG image bytes
    """
    from PIL import Image, ImageDraw

    if not page_patches.page_image_path:
        raise ValueError("Page image path not available")

    # Load page image
    img = Image.open(page_patches.page_image_path).convert("RGBA")

    # Create overlay for highlights
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw highlight rectangles
    highlight_ids = {p.patch_id for p in patches_to_highlight}
    for patch in page_patches.patches:
        if patch.patch_id in highlight_ids:
            x1 = patch.pixel_x
            y1 = patch.pixel_y
            x2 = patch.pixel_x + patch.pixel_width
            y2 = patch.pixel_y + patch.pixel_height

            # Draw filled rectangle with alpha
            alpha = int(255 * highlight_alpha)
            fill_color = (*highlight_color, alpha)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)

            # Draw border
            border_color = (*highlight_color, 255)
            draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)

    # Composite overlay onto image
    result = Image.alpha_composite(img, overlay)

    # Convert to PNG bytes
    buffer = io.BytesIO()
    result.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
