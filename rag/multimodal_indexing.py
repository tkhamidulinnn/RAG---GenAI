"""
Multimodal Indexing for RAG (Practice 5)

Creates and manages separate indices for different modalities:
- Text chunks (existing)
- Table representations
- Image descriptions

Supports unified retrieval across modalities with modality-aware filtering.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag.embeddings import GeminiEmbeddings
from rag.table_extraction import ExtractedTable, load_tables
from rag.image_extraction import ExtractedImage, load_images


# Phrases that indicate contact/logo/signatures/boilerplate (not substantive chart content)
_BOILERPLATE_PHRASES = (
    "[BOILERPLATE",
    "contact information",
    "contact details",
    "auditor",
    "deloitte",
    "touche",
    "tel:", "fax:", "tel ", "fax ",
    "www.", ".com",
    "suite ", "place", "mclean", "address",
    "independent auditors",
    "legal disclaimer",
    "signatures of",
    "signature",
    "executive",
    "president",
    "managing director",
    "names and titles",
    "key executives",
)
# Short generic descriptions (VLM didn't describe chart content)
_BOILERPLATE_GENERIC = ("image from financial report", "image from page")
# Words that suggest real chart content (do NOT include "figure" — it appears in "[FIGURE] Page N, figure_005")
_SUBSTANTIVE_HINTS = ("chart", "graph", "trend", "data", "axis", "value", "percent", "growth", "comparison", "distribution", "climate", "investment", "sustainability")


def _is_image_boilerplate(description: str) -> bool:
    """True if the image description looks like contact/logo/generic, not substantive content."""
    if not description:
        return False
    lower = description.lower()
    if any(phrase.lower() in lower for phrase in _BOILERPLATE_PHRASES):
        return True
    # Very short generic description (e.g. "Image from financial report") with no chart-like words
    if len(description) < 150 and any(g in lower for g in _BOILERPLATE_GENERIC):
        if not any(h in lower for h in _SUBSTANTIVE_HINTS):
            return True
    return False


@dataclass
class MultimodalDocument:
    """Document with modality information."""
    content: str
    modality: str  # "text", "table", "image"
    page: int
    item_id: str  # chunk_id, table_id, or figure_id
    metadata: Dict[str, Any]

    def to_langchain_doc(self) -> Document:
        """Convert to LangChain Document."""
        meta = {
            "modality": self.modality,
            "page": self.page,
            "item_id": self.item_id,
            **self.metadata,
        }
        return Document(page_content=self.content, metadata=meta)


def tables_to_documents(tables: List[ExtractedTable]) -> List[Document]:
    """Convert extracted tables to LangChain Documents for indexing."""
    documents = []

    for table in tables:
        # Create document from text representation
        doc = Document(
            page_content=table.text_representation,
            metadata={
                "modality": "table",
                "page": table.metadata.page,
                "item_id": table.metadata.table_id,
                "table_id": table.metadata.table_id,
                "caption": table.metadata.caption or "",
                "row_count": table.metadata.row_count,
                "col_count": table.metadata.col_count,
                "has_headers": table.metadata.has_headers,
                "source": "pdf_table",
            }
        )
        documents.append(doc)

    return documents


def images_to_documents(images: List[ExtractedImage]) -> List[Document]:
    """Convert extracted images to LangChain Documents for indexing."""
    documents = []

    for image in images:
        fp = image.metadata.file_path or ""
        doc = Document(
            page_content=image.description,
            metadata={
                "modality": "image",
                "page": image.metadata.page,
                "page_number": image.metadata.page,
                "item_id": image.metadata.figure_id,
                "figure_id": image.metadata.figure_id,
                "image_id": image.metadata.figure_id,
                "caption": image.metadata.caption or "",
                "image_type": image.metadata.image_type,
                "chart_type": image.metadata.chart_type or "",
                "file_path": fp,
                "image_path": fp,
                "source": "pdf_image",
            }
        )
        documents.append(doc)

    return documents


def build_table_index(
    tables: List[ExtractedTable],
    embeddings: GeminiEmbeddings,
    output_dir: Path,
) -> Optional[FAISS]:
    """Build FAISS index for tables."""
    documents = tables_to_documents(tables)

    if not documents:
        return None

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(str(output_dir))
    return db


def build_image_index(
    images: List[ExtractedImage],
    embeddings: GeminiEmbeddings,
    output_dir: Path,
) -> Optional[FAISS]:
    """Build FAISS index for images."""
    documents = images_to_documents(images)

    if not documents:
        return None

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(str(output_dir))
    return db


def load_table_index(
    index_dir: Path,
    embeddings: GeminiEmbeddings,
) -> Optional[FAISS]:
    """Load table index from disk."""
    if not index_dir.exists():
        return None

    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_image_index(
    index_dir: Path,
    embeddings: GeminiEmbeddings,
) -> Optional[FAISS]:
    """Load image index from disk."""
    if not index_dir.exists():
        return None

    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


class MultimodalRetriever:
    """
    Retriever that searches across multiple modalities.

    Supports:
    - Text only (baseline)
    - Tables only
    - Images only
    - Combined (text + tables + images)
    """

    # Keywords suggesting table-related queries
    TABLE_KEYWORDS = [
        "table", "row", "column", "cell", "value", "total", "sum",
        "percent", "percentage", "%", "fy", "fiscal", "quarter",
        "revenue", "income", "expense", "cost", "profit", "loss",
        "billion", "million", "thousand", "$", "usd", "eur",
        "increase", "decrease", "growth", "change", "difference",
        "compare", "comparison", "ratio", "margin", "rate",
    ]

    # Keywords suggesting image/chart-related queries
    IMAGE_KEYWORDS = [
        "chart", "graph", "figure", "diagram", "plot", "visualization",
        "trend", "line", "bar", "pie", "area", "scatter",
        "show", "display", "illustrate", "depict", "image", "picture",
        "climate", "sustainability", "environment", "investment", "investments",
        "net income", "income", "revenue", "fy22", "fy23", "fy24", "fy20",
        "over time", "across years", "historical",
    ]

    def __init__(
        self,
        text_index: Optional[FAISS] = None,
        table_index: Optional[FAISS] = None,
        image_index: Optional[FAISS] = None,
        tables_data: Optional[List[ExtractedTable]] = None,
        images_data: Optional[List[ExtractedImage]] = None,
    ):
        self.text_index = text_index
        self.table_index = table_index
        self.image_index = image_index
        self.tables_data = {t.metadata.table_id: t for t in (tables_data or [])}
        self.images_data = {i.metadata.figure_id: i for i in (images_data or [])}

    def _detect_query_intent(self, query: str) -> Dict[str, float]:
        """
        Detect query intent based on keywords.

        Returns weights for each modality.
        """
        query_lower = query.lower()

        # Count keyword matches
        table_score = sum(1 for kw in self.TABLE_KEYWORDS if kw in query_lower)
        image_score = sum(1 for kw in self.IMAGE_KEYWORDS if kw in query_lower)

        # Normalize to weights
        total = table_score + image_score + 1  # +1 for base text weight

        if table_score == 0 and image_score == 0:
            # No specific intent detected, use balanced weights
            return {"text": 0.6, "table": 0.25, "image": 0.15}

        # Weighted based on detected intent
        weights = {
            "text": 0.3,
            "table": 0.4 * (table_score / total) + 0.1,
            "image": 0.4 * (image_score / total) + 0.1,
        }

        # Normalize
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}

    def retrieve(
        self,
        query: str,
        mode: str = "auto",  # "text", "tables", "images", "mixed", "auto"
        top_k_text: int = 5,
        top_k_tables: int = 3,
        top_k_images: int = 2,
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents across modalities.

        Args:
            query: Search query
            mode: Retrieval mode
            top_k_text: Number of text chunks to retrieve
            top_k_tables: Number of tables to retrieve
            top_k_images: Number of images to retrieve

        Returns:
            Dict with keys "text", "tables", "images" containing Document lists
        """
        results = {
            "text": [],
            "tables": [],
            "images": [],
        }

        # Determine which indices to query based on mode
        if mode == "auto":
            weights = self._detect_query_intent(query)
            query_text = weights["text"] > 0.1 and self.text_index is not None
            query_tables = weights["table"] > 0.1 and self.table_index is not None
            query_images = weights["image"] > 0.1 and self.image_index is not None
        elif mode == "text":
            query_text = self.text_index is not None
            query_tables = False
            query_images = False
        elif mode == "tables":
            query_text = False
            query_tables = self.table_index is not None
            query_images = False
        elif mode == "images":
            query_text = False
            query_tables = False
            query_images = self.image_index is not None
        else:  # mixed
            query_text = self.text_index is not None
            query_tables = self.table_index is not None
            query_images = self.image_index is not None

        # Retrieve from each index
        if query_text:
            results["text"] = self.text_index.similarity_search(query, k=top_k_text)

        if query_tables:
            results["tables"] = self.table_index.similarity_search(query, k=top_k_tables)

        if query_images:
            fetch_k = min(top_k_images + 10, 20)
            image_docs = self.image_index.similarity_search(query, k=fetch_k)
            substantive = [d for d in image_docs if not _is_image_boilerplate(d.page_content or "")]
            boilerplate = [d for d in image_docs if _is_image_boilerplate(d.page_content or "")]
            query_lower = query.lower()
            thematic = any(
                w in query_lower for w in ("climate", "sustainability", "investment", "investments", "environment", "green")
            )
            if thematic:
                # Prefer images that mention the theme in description; if none, show best substantive (no contact/signatures)
                theme_words = [w for w in ("climate", "sustainability", "investment", "environment", "green", "paris agreement", "finance") if w in query_lower]
                theme_matching = substantive
                if theme_words:
                    def _matches_theme(doc: Document) -> bool:
                        content = (doc.page_content or "").lower()
                        return any(t in content for t in theme_words)
                    theme_matching = [d for d in substantive if _matches_theme(d)]
                # Use theme-matching if any, else fall back to all substantive so we still show some images
                results["images"] = (theme_matching[:top_k_images] if theme_matching else substantive[:top_k_images])
            else:
                reordered = substantive + boilerplate
                results["images"] = reordered[:top_k_images]

        return results

    def get_table_data(self, table_id: str) -> Optional[ExtractedTable]:
        """Get full table data by ID."""
        return self.tables_data.get(table_id)

    def get_image_data(self, figure_id: str) -> Optional[ExtractedImage]:
        """Get full image data by ID."""
        return self.images_data.get(figure_id)


def create_multimodal_retriever(
    text_index_dir: Optional[Path] = None,
    table_index_dir: Optional[Path] = None,
    image_index_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    embeddings: Optional[GeminiEmbeddings] = None,
) -> MultimodalRetriever:
    """
    Factory function to create MultimodalRetriever.

    Args:
        text_index_dir: Directory with text FAISS index
        table_index_dir: Directory with table FAISS index
        image_index_dir: Directory with image FAISS index
        tables_dir: Directory with table JSON files
        images_dir: Directory with image JSON files
        embeddings: Embeddings model

    Returns:
        Configured MultimodalRetriever
    """
    text_index = None
    table_index = None
    image_index = None
    tables_data = []
    images_data = []

    if embeddings:
        if text_index_dir and text_index_dir.exists():
            text_index = FAISS.load_local(
                str(text_index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )

        if table_index_dir and table_index_dir.exists():
            table_index = load_table_index(table_index_dir, embeddings)

        if image_index_dir and image_index_dir.exists():
            image_index = load_image_index(image_index_dir, embeddings)

    if tables_dir and tables_dir.exists():
        tables_data = load_tables(tables_dir)

    if images_dir and images_dir.exists():
        images_data = load_images(images_dir)

    return MultimodalRetriever(
        text_index=text_index,
        table_index=table_index,
        image_index=image_index,
        tables_data=tables_data,
        images_data=images_data,
    )
