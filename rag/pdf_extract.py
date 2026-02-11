from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
try:
    from google.genai import types
except ImportError as exc:
    raise RuntimeError(
        "google-genai not found. Run: pip install -r requirements.txt"
    ) from exc

from rag.gemini_client import generate_json


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def extract_text(pdf_path: Path, artifacts_dir: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    text_rows: List[Dict[str, Any]] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text = page.get_text("text")
        text_rows.append(
            {
                "page": page_index + 1,
                "text": text,
                "width": page.rect.width,
                "height": page.rect.height,
            }
        )
    _write_jsonl(artifacts_dir / "text" / "pages.jsonl", text_rows)
    return text_rows


def extract_images(
    pdf_path: Path,
    artifacts_dir: Path,
    gemini_model: str,
    client,
) -> List[Dict[str, Any]]:
    """
    Extract images from PDF with basic captioning (legacy function).
    
    NOTE: This is a legacy function used only by extract_all() for basic text extraction.
    For multimodal RAG (Practice 5), use rag.image_extraction.extract_images_from_pdf() instead,
    which provides VLM-based descriptions and chart type detection.
    """
    doc = fitz.open(pdf_path)
    image_rows: List[Dict[str, Any]] = []
    images_dir = artifacts_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            extracted = doc.extract_image(xref)
            ext = extracted.get("ext", "png")
            image_bytes = extracted.get("image")
            if not image_bytes:
                continue
            filename = f"page_{page_index + 1:03d}_img_{image_index:03d}.{ext}"
            image_path = images_dir / filename
            image_path.write_bytes(image_bytes)

            prompt = (
                "You are captioning a PDF image for retrieval. "
                "Return a short, concrete caption, key objects, and a guess of the image role."
            )
            response_schema = {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "role": {"type": "string"},
                },
                "required": ["caption", "objects", "role"],
            }
            parts = [types.Part.from_bytes(data=image_bytes, mime_type=f"image/{ext}")]
            try:
                caption_payload = generate_json(
                    client=client,
                    model=gemini_model,
                    prompt=prompt,
                    response_schema=response_schema,
                    parts=parts,
                )
            except Exception as exc:
                caption_payload = {"caption": f"caption_failed: {exc}", "objects": [], "role": "unknown"}

            image_rows.append(
                {
                    "page": page_index + 1,
                    "xref": xref,
                    "file_name": filename,
                    "width": extracted.get("width"),
                    "height": extracted.get("height"),
                    "caption": caption_payload,
                }
            )

    _write_jsonl(artifacts_dir / "images" / "metadata.jsonl", image_rows)
    return image_rows


def extract_tables(pdf_path: Path, artifacts_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract tables from PDF using camelot (legacy function).
    
    NOTE: This is a legacy function used only by extract_all() for basic text extraction.
    For multimodal RAG (Practice 5), use rag.table_extraction.extract_tables() instead,
    which provides dual representations and better structure preservation.
    
    Requires optional dependency: camelot-py[cv] (not in requirements.txt)
    """
    try:
        import camelot
    except Exception:
        return []

    tables_dir = artifacts_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables = camelot.read_pdf(str(pdf_path), pages="all")
    table_rows: List[Dict[str, Any]] = []
    for index, table in enumerate(tables):
        table_path = tables_dir / f"table_{index + 1:03d}.csv"
        table.df.to_csv(table_path, index=False)
        table_rows.append(
            {
                "index": index + 1,
                "page": table.page,
                "shape": table.shape,
                "accuracy": getattr(table, "accuracy", None),
                "file_name": table_path.name,
            }
        )
    _write_jsonl(artifacts_dir / "tables" / "metadata.jsonl", table_rows)
    return table_rows


def extract_all(
    pdf_path: Path,
    artifacts_dir: Path,
    gemini_model: str,
    client,
    skip_images: bool = False,
    skip_tables: bool = False,
) -> Dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pdf_path": str(pdf_path),
    }

    doc = fitz.open(pdf_path)
    metadata.update(doc.metadata or {})
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    text_rows = extract_text(pdf_path, artifacts_dir)
    image_rows = [] if skip_images else extract_images(pdf_path, artifacts_dir, gemini_model, client)
    table_rows = [] if skip_tables else extract_tables(pdf_path, artifacts_dir)

    return {
        "text_pages": len(text_rows),
        "images": len(image_rows),
        "tables": len(table_rows),
    }
