"""
Metadata Extraction for Enhanced Retrieval (Practice 3)

Extracts structured metadata from PDF pages to enable:
- Section detection (Executive Summary, Financial Highlights, Notes, etc.)
- Content type classification (text, table, header, figure)
- Page-level metadata for filtering

This enables metadata-based filtering at retrieval time.

References:
- Anthropic: Contextual Retrieval
- Qdrant: Payload Filtering
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# Common section patterns in financial reports
SECTION_PATTERNS = [
    # Executive sections
    (r"executive\s+summary", "Executive Summary"),
    (r"letter\s+(?:from|to)\s+(?:the\s+)?(?:president|ceo|chairman|shareholders)", "Letter to Stakeholders"),
    (r"message\s+from\s+(?:the\s+)?(?:president|ceo|chairman)", "Letter to Stakeholders"),

    # Financial sections
    (r"financial\s+highlights", "Financial Highlights"),
    (r"financial\s+statements?", "Financial Statements"),
    (r"consolidated\s+(?:financial\s+)?statements?", "Financial Statements"),
    (r"balance\s+sheet", "Balance Sheet"),
    (r"income\s+statement", "Income Statement"),
    (r"statement\s+of\s+(?:comprehensive\s+)?income", "Income Statement"),
    (r"cash\s+flow", "Cash Flow Statement"),
    (r"statement\s+of\s+cash\s+flows?", "Cash Flow Statement"),
    (r"statement\s+of\s+changes\s+in\s+equity", "Equity Statement"),

    # Notes and disclosures
    (r"notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements?", "Notes to Financial Statements"),
    (r"significant\s+accounting\s+policies", "Accounting Policies"),
    (r"critical\s+accounting\s+(?:policies|estimates)", "Accounting Policies"),
    (r"risk\s+management", "Risk Management"),
    (r"credit\s+risk", "Risk Management"),
    (r"market\s+risk", "Risk Management"),
    (r"liquidity\s+risk", "Risk Management"),

    # Governance
    (r"corporate\s+governance", "Corporate Governance"),
    (r"board\s+of\s+directors?", "Corporate Governance"),
    (r"management['']?s?\s+discussion", "Management Discussion"),
    (r"md&a|management\s+discussion\s+and\s+analysis", "Management Discussion"),

    # Other common sections
    (r"auditor['']?s?\s+report", "Auditor's Report"),
    (r"independent\s+auditor", "Auditor's Report"),
    (r"appendix|annex", "Appendix"),
    (r"glossary", "Glossary"),
    (r"table\s+of\s+contents", "Table of Contents"),
]


@dataclass
class PageMetadata:
    """Metadata extracted from a single page."""
    page_number: int
    section: str  # Detected section name
    content_type: str  # "text", "table", "header", "mixed"
    has_tables: bool
    has_figures: bool
    word_count: int
    line_count: int
    detected_headers: List[str]  # Potential section headers found


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk (after splitting)."""
    page: int
    section: str
    content_type: str
    chunk_index: int  # Index within the page
    word_count: int
    has_numbers: bool  # Contains financial numbers
    has_dates: bool  # Contains dates


def detect_section(text: str, previous_section: str = "Introduction") -> str:
    """Detect the section name from text content.

    Uses pattern matching to identify common financial report sections.
    Falls back to the previous section if no match is found.
    """
    text_lower = text.lower()

    # Check first 500 chars for section headers
    header_region = text_lower[:500]

    for pattern, section_name in SECTION_PATTERNS:
        if re.search(pattern, header_region):
            return section_name

    return previous_section


def detect_content_type(text: str) -> str:
    """Classify content type based on text structure.

    Returns:
        "header": Short text, likely a heading
        "table": Contains tabular structure
        "text": Regular paragraph text
        "mixed": Combination of types
    """
    lines = text.strip().split("\n")
    line_count = len(lines)

    if line_count == 0:
        return "text"

    # Check for table-like structure (many short lines with numbers)
    numeric_lines = sum(1 for line in lines if _has_table_structure(line))
    numeric_ratio = numeric_lines / line_count

    # Header detection: short text, possibly all caps
    avg_line_length = sum(len(line) for line in lines) / line_count
    if line_count <= 3 and avg_line_length < 100:
        if text.strip().isupper() or _looks_like_header(text):
            return "header"

    # Table detection
    if numeric_ratio > 0.5 and line_count > 3:
        return "table"

    if numeric_ratio > 0.2:
        return "mixed"

    return "text"


def _has_table_structure(line: str) -> bool:
    """Check if a line looks like a table row."""
    # Multiple numbers separated by whitespace
    numbers = re.findall(r"[\d,]+\.?\d*", line)
    if len(numbers) >= 2:
        return True

    # Tab or multiple space separated values
    if "\t" in line or "  " in line:
        parts = re.split(r"\t|\s{2,}", line.strip())
        if len(parts) >= 3:
            return True

    return False


def _looks_like_header(text: str) -> bool:
    """Check if text looks like a section header."""
    text = text.strip()

    # All caps
    if text.isupper() and len(text) < 100:
        return True

    # Title case with few words
    words = text.split()
    if len(words) <= 6:
        title_case_words = sum(1 for w in words if w[0].isupper() if w)
        if title_case_words / len(words) > 0.7:
            return True

    return False


def extract_headers(text: str) -> List[str]:
    """Extract potential section headers from text."""
    headers = []
    lines = text.split("\n")

    for line in lines[:20]:  # Check first 20 lines
        line = line.strip()
        if not line:
            continue

        # Short lines that are all caps or title case
        if len(line) < 100 and _looks_like_header(line):
            headers.append(line)

    return headers[:5]  # Return at most 5 headers


def extract_page_metadata(
    page_text: str,
    page_number: int,
    previous_section: str = "Introduction",
) -> PageMetadata:
    """Extract metadata from a single page.

    Args:
        page_text: Full text content of the page
        page_number: 1-indexed page number
        previous_section: Section from the previous page (for continuity)

    Returns:
        PageMetadata with detected section, content type, etc.
    """
    section = detect_section(page_text, previous_section)
    content_type = detect_content_type(page_text)
    headers = extract_headers(page_text)

    # Count lines and words
    lines = page_text.strip().split("\n")
    words = page_text.split()

    # Check for tables and figures
    has_tables = content_type in ("table", "mixed")
    has_figures = bool(re.search(r"figure\s+\d|chart|graph|illustration", page_text.lower()))

    return PageMetadata(
        page_number=page_number,
        section=section,
        content_type=content_type,
        has_tables=has_tables,
        has_figures=has_figures,
        word_count=len(words),
        line_count=len(lines),
        detected_headers=headers,
    )


def extract_chunk_metadata(
    chunk_text: str,
    page_number: int,
    section: str,
    chunk_index: int,
) -> ChunkMetadata:
    """Extract metadata for a text chunk.

    Args:
        chunk_text: The chunk content
        page_number: Page the chunk came from
        section: Section name (from page metadata)
        chunk_index: Index of this chunk within the page

    Returns:
        ChunkMetadata for the chunk
    """
    content_type = detect_content_type(chunk_text)
    words = chunk_text.split()

    # Check for financial numbers (currency, percentages, large numbers)
    has_numbers = bool(re.search(
        r"\$[\d,]+|\d+%|[\d,]+\.\d{2}|\b\d{1,3}(?:,\d{3})+\b",
        chunk_text
    ))

    # Check for dates
    has_dates = bool(re.search(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b"
        r"|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"
        r"|\b(?:fy|fiscal\s+year)\s*\d{2,4}\b",
        chunk_text,
        re.IGNORECASE
    ))

    return ChunkMetadata(
        page=page_number,
        section=section,
        content_type=content_type,
        chunk_index=chunk_index,
        word_count=len(words),
        has_numbers=has_numbers,
        has_dates=has_dates,
    )


def enrich_document_metadata(
    documents: List,
    page_metadata_map: Dict[int, PageMetadata],
) -> List:
    """Enrich LangChain documents with extracted metadata.

    Args:
        documents: List of LangChain Document objects (after chunking)
        page_metadata_map: Mapping from page number to PageMetadata

    Returns:
        Documents with enriched metadata
    """
    chunk_counts: Dict[int, int] = {}  # Track chunk index per page

    for doc in documents:
        page = doc.metadata.get("page", 1)

        # Get or create page metadata
        if page in page_metadata_map:
            page_meta = page_metadata_map[page]
            section = page_meta.section
        else:
            section = "Unknown"

        # Get chunk index for this page
        chunk_index = chunk_counts.get(page, 0)
        chunk_counts[page] = chunk_index + 1

        # Extract chunk-specific metadata
        chunk_meta = extract_chunk_metadata(
            doc.page_content,
            page,
            section,
            chunk_index,
        )

        # Update document metadata
        doc.metadata.update({
            "section": chunk_meta.section,
            "content_type": chunk_meta.content_type,
            "chunk_index": chunk_meta.chunk_index,
            "has_numbers": chunk_meta.has_numbers,
            "has_dates": chunk_meta.has_dates,
        })

    return documents


def build_page_metadata_map(pages: List[Dict]) -> Dict[int, PageMetadata]:
    """Build a mapping of page numbers to metadata.

    Args:
        pages: List of page dicts with "page" and "text" keys

    Returns:
        Dict mapping page number to PageMetadata
    """
    metadata_map = {}
    previous_section = "Introduction"

    for page_data in pages:
        page_num = page_data.get("page", 1)
        text = page_data.get("text", "")

        page_meta = extract_page_metadata(text, page_num, previous_section)
        metadata_map[page_num] = page_meta
        previous_section = page_meta.section

    return metadata_map
