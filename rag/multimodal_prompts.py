"""
Multimodal QA Prompts for RAG (Practice 5)

Specialized prompts for:
- Table QA: Answering questions using tabular data
- Visual QA: Answering questions using chart/graph descriptions
- Combined QA: Answering using text + tables + images
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Iterator

from langchain_core.documents import Document

from rag.gemini_client import stream_text, generate_json


# =============================================================================
# TABLE QA PROMPT
# =============================================================================

TABLE_QA_SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions using ONLY the provided table data.

CRITICAL RULES:
1. ONLY use information from the provided tables. Do NOT use prior knowledge.
2. If the answer is not in the tables, say "The provided tables do not contain this information."
3. Always cite your evidence: page number, table ID, and specific cell references.
4. For calculations, show your work with the exact values from the table.

When answering:
- First identify which table(s) contain relevant data
- Extract the exact cell values needed
- If calculation is required, show: value1 (row X, col Y) + value2 (row X, col Z) = result
- Format numbers consistently with the source"""


def build_table_qa_prompt(
    query: str,
    table_docs: List[Document],
) -> str:
    """Build prompt for table-based QA."""
    if not table_docs:
        return f"""Question: {query}

No tables were retrieved. Please rephrase your question or check if the document contains relevant tabular data.

Answer: I cannot answer this question as no relevant tables were found."""

    # Build table context
    table_context = []
    for doc in table_docs:
        page = doc.metadata.get("page", "?")
        table_id = doc.metadata.get("table_id", "unknown")
        caption = doc.metadata.get("caption", "")

        header = f"[Table: {table_id}, Page {page}]"
        if caption:
            header += f"\nCaption: {caption}"

        table_context.append(f"{header}\n{doc.page_content}")

    tables_text = "\n\n---\n\n".join(table_context)

    return f"""{TABLE_QA_SYSTEM_PROMPT}

TABLES:
{tables_text}

Question: {query}

Answer the question using ONLY the data from the tables above. Include:
1. Your answer
2. Evidence: specific table ID, page, and cell references used
3. If you performed a calculation, show the formula with actual values

Answer:"""


# =============================================================================
# VISUAL QA PROMPT
# =============================================================================

VISUAL_QA_SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions using ONLY the provided chart/graph descriptions.

CRITICAL RULES:
1. ONLY use information from the provided image descriptions. Do NOT use prior knowledge.
2. If the answer is not in the descriptions, say "The provided figures do not contain this information."
3. Always cite your evidence: page number, figure ID, and specific data points.
4. Be precise about values - only report what is explicitly stated in the description.

When answering:
- Identify which figure(s) contain relevant information
- Extract specific values, trends, or patterns mentioned
- Note any limitations (e.g., "The chart shows a trend but exact values are not visible")"""


def build_visual_qa_prompt(
    query: str,
    image_docs: List[Document],
) -> str:
    """Build prompt for visual/chart-based QA."""
    if not image_docs:
        return f"""Question: {query}

No figures/charts were retrieved. Please rephrase your question or check if the document contains relevant visual data.

Answer: I cannot answer this question as no relevant figures were found."""

    # Build image context
    image_context = []
    for doc in image_docs:
        page = doc.metadata.get("page", "?")
        figure_id = doc.metadata.get("figure_id", "unknown")
        chart_type = doc.metadata.get("chart_type", "")
        caption = doc.metadata.get("caption", "")

        header = f"[Figure: {figure_id}, Page {page}]"
        if chart_type:
            header += f" (Type: {chart_type})"
        if caption:
            header += f"\nCaption: {caption}"

        image_context.append(f"{header}\n{doc.page_content}")

    images_text = "\n\n---\n\n".join(image_context)

    return f"""{VISUAL_QA_SYSTEM_PROMPT}

FIGURES/CHARTS:
{images_text}

Question: {query}

Answer the question using ONLY the figure descriptions above. Include:
1. Your answer
2. Evidence: specific figure ID, page, and data points referenced
3. Note any limitations in the available data

Answer:"""


# =============================================================================
# COMBINED MULTIMODAL QA PROMPT
# =============================================================================

MULTIMODAL_QA_SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions using the provided context which may include text, tables, and figure descriptions.

CRITICAL RULES:
1. ONLY use information from the provided context. Do NOT use prior knowledge.
2. If the answer is not in the context, say "The provided context does not contain this information."
3. Always cite your evidence with specific references:
   - For text: [page X]
   - For tables: [Table table_id, page X, row/column reference]
   - For figures: [Figure figure_id, page X]
4. Prefer tabular data for precise numeric values
5. Prefer text for explanations and context
6. Prefer figures for trends and visual patterns"""


def build_multimodal_qa_prompt(
    query: str,
    text_docs: List[Document],
    table_docs: List[Document],
    image_docs: List[Document],
) -> str:
    """Build prompt for combined multimodal QA."""
    context_parts = []

    # Add text context
    if text_docs:
        text_items = []
        for doc in text_docs:
            page = doc.metadata.get("page", "?")
            text_items.append(f"[Page {page}]\n{doc.page_content}")
        context_parts.append("TEXT EXCERPTS:\n" + "\n\n".join(text_items))

    # Add table context
    if table_docs:
        table_items = []
        for doc in table_docs:
            page = doc.metadata.get("page", "?")
            table_id = doc.metadata.get("table_id", "unknown")
            caption = doc.metadata.get("caption", "")
            header = f"[Table {table_id}, Page {page}]"
            if caption:
                header += f" Caption: {caption}"
            table_items.append(f"{header}\n{doc.page_content}")
        context_parts.append("TABLES:\n" + "\n\n".join(table_items))

    # Add image context
    if image_docs:
        image_items = []
        for doc in image_docs:
            page = doc.metadata.get("page", "?")
            figure_id = doc.metadata.get("figure_id", "unknown")
            chart_type = doc.metadata.get("chart_type", "")
            caption = doc.metadata.get("caption", "")
            header = f"[Figure {figure_id}, Page {page}]"
            if chart_type:
                header += f" (Type: {chart_type})"
            if caption:
                header += f" Caption: {caption}"
            image_items.append(f"{header}\n{doc.page_content}")
        context_parts.append("FIGURES/CHARTS:\n" + "\n\n".join(image_items))

    if not context_parts:
        return f"""Question: {query}

No relevant context was found. Please rephrase your question.

Answer: I cannot answer this question as no relevant information was found."""

    full_context = "\n\n" + "=" * 50 + "\n\n".join(context_parts)

    return f"""{MULTIMODAL_QA_SYSTEM_PROMPT}

CONTEXT:
{full_context}

Question: {query}

Provide a comprehensive answer using the context above. Include:
1. Your answer
2. Evidence citations (page, table ID, figure ID as applicable)
3. Any calculations with source values shown

Answer:"""


# =============================================================================
# STREAMING AND STRUCTURED RESPONSES
# =============================================================================

def answer_table_qa_stream(
    client,
    model: str,
    query: str,
    table_docs: List[Document],
) -> Iterator[str]:
    """Stream answer for table QA."""
    prompt = build_table_qa_prompt(query, table_docs)
    return stream_text(client=client, model=model, prompt=prompt)


def answer_visual_qa_stream(
    client,
    model: str,
    query: str,
    image_docs: List[Document],
) -> Iterator[str]:
    """Stream answer for visual QA."""
    prompt = build_visual_qa_prompt(query, image_docs)
    return stream_text(client=client, model=model, prompt=prompt)


def answer_multimodal_qa_stream(
    client,
    model: str,
    query: str,
    text_docs: List[Document],
    table_docs: List[Document],
    image_docs: List[Document],
) -> Iterator[str]:
    """Stream answer for multimodal QA."""
    prompt = build_multimodal_qa_prompt(query, text_docs, table_docs, image_docs)
    return stream_text(client=client, model=model, prompt=prompt)


def answer_multimodal_structured(
    client,
    model: str,
    query: str,
    text_docs: List[Document],
    table_docs: List[Document],
    image_docs: List[Document],
) -> Dict[str, Any]:
    """Generate structured multimodal answer."""
    prompt = build_multimodal_qa_prompt(query, text_docs, table_docs, image_docs)

    response_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_type": {"type": "string"},  # text, table, figure
                        "source_id": {"type": "string"},  # page, table_id, figure_id
                        "page": {"type": "integer"},
                        "reference": {"type": "string"},  # specific cell/row/value
                    },
                },
            },
            "calculation": {
                "type": "object",
                "properties": {
                    "formula": {"type": "string"},
                    "inputs": {"type": "object"},
                    "result": {"type": "string"},
                },
            },
            "confidence": {"type": "string"},  # high, medium, low
            "modalities_used": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["answer", "citations"],
    }

    try:
        return generate_json(
            client=client,
            model=model,
            prompt=prompt,
            response_schema=response_schema,
        )
    except Exception:
        # Fallback
        return {
            "answer": "Error generating structured response",
            "citations": [],
        }
