"""
Table Extraction and Processing for Multimodal RAG (Practice 5.1)

Extracts tables from PDFs and creates dual representations:
- R1: Text representation for retrieval (markdown/CSV format)
- R2: Structured JSON representation for computation

Features:
- Multiple extraction strategies (PyMuPDF, pdfplumber fallback)
- Caption detection from surrounding text
- Numeric normalization for financial data
- Metadata preservation (page, table_id, headers, shape)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import io


@dataclass
class TableMetadata:
    """Metadata for an extracted table."""
    doc_id: str
    page: int
    table_id: str
    caption: Optional[str] = None
    extraction_method: str = "pymupdf"
    row_count: int = 0
    col_count: int = 0
    has_headers: bool = True
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ExtractedTable:
    """Complete table representation with both retrieval and structured formats."""
    metadata: TableMetadata
    # R1: Text representation for retrieval
    text_representation: str
    # R2: Structured representation for computation
    structured_data: Dict[str, Any]
    # Raw data
    raw_rows: List[List[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "text_representation": self.text_representation,
            "structured_data": self.structured_data,
            "raw_rows": self.raw_rows,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedTable":
        return cls(
            metadata=TableMetadata(**data["metadata"]),
            text_representation=data["text_representation"],
            structured_data=data["structured_data"],
            raw_rows=data.get("raw_rows", []),
        )


def _normalize_numeric(value: str) -> Optional[float]:
    """Normalize numeric values from financial tables."""
    if not value or not value.strip():
        return None

    cleaned = value.strip()

    # Handle parentheses for negative numbers
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    # Remove currency symbols and commas
    cleaned = re.sub(r"[$€£¥,\s]", "", cleaned)

    # Handle percentages
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]

    # Handle dash or em-dash as zero or missing
    if cleaned in ["-", "—", "–", "N/A", "n/a", ""]:
        return None

    try:
        result = float(cleaned)
        if is_percent:
            result = result / 100
        return result
    except ValueError:
        return None


def _detect_caption(page_text: str, table_bbox: Optional[Tuple], page_height: float) -> Optional[str]:
    """Detect table caption from surrounding page text."""
    # Look for common caption patterns
    patterns = [
        r"Table\s+\d+[:\.]?\s*([^\n]+)",
        r"TABLE\s+\d+[:\.]?\s*([^\n]+)",
        r"Exhibit\s+\d+[:\.]?\s*([^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, page_text)
        if match:
            caption = match.group(0).strip()
            # Limit caption length
            if len(caption) > 200:
                caption = caption[:200] + "..."
            return caption

    return None


def _create_text_representation(
    rows: List[List[str]],
    metadata: TableMetadata,
) -> str:
    """Create R1: text representation for retrieval."""
    lines = []

    # Header
    lines.append(f"[TABLE] Page {metadata.page}, {metadata.table_id}")
    if metadata.caption:
        lines.append(f"Caption: {metadata.caption}")

    # Convert to markdown table format
    if not rows:
        return "\n".join(lines)

    # Determine column widths for alignment
    col_count = max(len(row) for row in rows)

    # Header row
    if metadata.has_headers and rows:
        header = rows[0]
        lines.append("| " + " | ".join(str(cell).strip() for cell in header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        data_rows = rows[1:]
    else:
        data_rows = rows

    # Data rows
    for row in data_rows:
        # Pad row to match column count
        padded = list(row) + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(str(cell).strip() for cell in padded) + " |")

    return "\n".join(lines)


def _create_structured_representation(
    rows: List[List[str]],
    metadata: TableMetadata,
) -> Dict[str, Any]:
    """Create R2: structured JSON representation for computation."""
    if not rows:
        return {"headers": [], "rows": [], "normalized": []}

    headers = rows[0] if metadata.has_headers else [f"col_{i}" for i in range(len(rows[0]))]
    data_rows = rows[1:] if metadata.has_headers else rows

    # Create structured rows
    structured_rows = []
    normalized_rows = []

    for row in data_rows:
        row_dict = {}
        normalized_dict = {}
        for i, cell in enumerate(row):
            if i < len(headers):
                key = str(headers[i]).strip() or f"col_{i}"
                row_dict[key] = str(cell).strip()
                normalized_dict[key] = _normalize_numeric(cell)
        structured_rows.append(row_dict)
        normalized_rows.append(normalized_dict)

    return {
        "headers": [str(h).strip() for h in headers],
        "rows": structured_rows,
        "normalized": normalized_rows,
        "row_count": len(data_rows),
        "col_count": len(headers),
    }


def extract_tables_pymupdf(
    pdf_path: Path,
    doc_id: str = "doc",
) -> List[ExtractedTable]:
    """Extract tables using PyMuPDF's built-in table detection."""
    import fitz

    doc = fitz.open(pdf_path)
    tables: List[ExtractedTable] = []
    global_table_idx = 0

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        page_text = page.get_text("text")

        # Try to find tables on the page
        try:
            page_tables = page.find_tables()
        except Exception:
            continue

        for table_idx, table in enumerate(page_tables):
            global_table_idx += 1
            table_id = f"table_{global_table_idx:03d}"

            # Extract table data
            try:
                raw_rows = table.extract()
            except Exception:
                continue

            if not raw_rows or len(raw_rows) < 2:
                continue

            # Clean empty rows
            raw_rows = [row for row in raw_rows if any(cell and str(cell).strip() for cell in row)]
            if not raw_rows:
                continue

            # Get table bbox
            try:
                bbox = table.bbox
            except Exception:
                bbox = None

            # Detect caption
            caption = _detect_caption(page_text, bbox, page.rect.height)

            # Create metadata
            metadata = TableMetadata(
                doc_id=doc_id,
                page=page_idx + 1,
                table_id=table_id,
                caption=caption,
                extraction_method="pymupdf",
                row_count=len(raw_rows),
                col_count=max(len(row) for row in raw_rows) if raw_rows else 0,
                has_headers=True,
                bbox=bbox,
            )

            # Create representations
            text_repr = _create_text_representation(raw_rows, metadata)
            structured = _create_structured_representation(raw_rows, metadata)

            tables.append(ExtractedTable(
                metadata=metadata,
                text_representation=text_repr,
                structured_data=structured,
                raw_rows=raw_rows,
            ))

    doc.close()
    return tables


def extract_tables_pdfplumber(
    pdf_path: Path,
    doc_id: str = "doc",
) -> List[ExtractedTable]:
    """Extract tables using pdfplumber as fallback."""
    try:
        import pdfplumber
    except ImportError:
        return []

    tables: List[ExtractedTable] = []
    global_table_idx = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            page_tables = page.extract_tables() or []

            for table_idx, raw_rows in enumerate(page_tables):
                if not raw_rows or len(raw_rows) < 2:
                    continue

                global_table_idx += 1
                table_id = f"table_{global_table_idx:03d}"

                # Clean rows
                raw_rows = [
                    [str(cell) if cell else "" for cell in row]
                    for row in raw_rows
                    if any(cell for cell in row)
                ]
                if not raw_rows:
                    continue

                # Detect caption
                caption = _detect_caption(page_text, None, page.height)

                metadata = TableMetadata(
                    doc_id=doc_id,
                    page=page_idx + 1,
                    table_id=table_id,
                    caption=caption,
                    extraction_method="pdfplumber",
                    row_count=len(raw_rows),
                    col_count=max(len(row) for row in raw_rows) if raw_rows else 0,
                    has_headers=True,
                )

                text_repr = _create_text_representation(raw_rows, metadata)
                structured = _create_structured_representation(raw_rows, metadata)

                tables.append(ExtractedTable(
                    metadata=metadata,
                    text_representation=text_repr,
                    structured_data=structured,
                    raw_rows=raw_rows,
                ))

    return tables


def extract_tables(
    pdf_path: Path,
    doc_id: str = "doc",
    method: str = "auto",
) -> List[ExtractedTable]:
    """
    Extract tables from PDF using specified method.

    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier
        method: "pymupdf", "pdfplumber", or "auto"

    Returns:
        List of ExtractedTable objects
    """
    if method == "pymupdf":
        return extract_tables_pymupdf(pdf_path, doc_id)
    elif method == "pdfplumber":
        return extract_tables_pdfplumber(pdf_path, doc_id)
    else:
        # Auto: try pymupdf first, fallback to pdfplumber
        tables = extract_tables_pymupdf(pdf_path, doc_id)
        if not tables:
            tables = extract_tables_pdfplumber(pdf_path, doc_id)
        return tables


def save_tables(tables: List[ExtractedTable], output_dir: Path) -> None:
    """Save extracted tables to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata and representations
    metadata_rows = []
    for table in tables:
        # Save individual table JSON
        table_file = output_dir / f"{table.metadata.table_id}.json"
        table_file.write_text(
            json.dumps(table.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metadata_rows.append({
            "table_id": table.metadata.table_id,
            "page": table.metadata.page,
            "caption": table.metadata.caption,
            "row_count": table.metadata.row_count,
            "col_count": table.metadata.col_count,
            "extraction_method": table.metadata.extraction_method,
        })

    # Save metadata index
    metadata_file = output_dir / "tables_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_tables(tables_dir: Path) -> List[ExtractedTable]:
    """Load extracted tables from disk."""
    tables = []
    metadata_file = tables_dir / "tables_metadata.json"

    if not metadata_file.exists():
        return tables

    with metadata_file.open("r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    for meta in metadata_list:
        table_file = tables_dir / f"{meta['table_id']}.json"
        if table_file.exists():
            with table_file.open("r", encoding="utf-8") as f:
                table_data = json.load(f)
                tables.append(ExtractedTable.from_dict(table_data))

    return tables
