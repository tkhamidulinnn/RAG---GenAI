from __future__ import annotations

import json
from typing import Iterator, List, Tuple

from google.genai import types

from rag.gemini_client import stream_text, generate_json, generate_with_function_call


def retrieve(
    db,
    query: str,
    k: int = 5,
) -> List:
    return db.similarity_search(query, k=k)


def build_prompt(query: str, docs: List) -> str:
    context_parts = []
    for doc in docs:
        page = doc.metadata.get("page")
        label = f"page {page}" if page else "page ?"
        context_parts.append(f"[{label}] {doc.page_content}")
    context = "\n\n".join(context_parts)
    return (
        "You are a financial report assistant. Use only the provided context.\n"
        "If the answer is not in the context, say you do not know.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def answer_stream(
    client,
    model: str,
    query: str,
    docs: List,
) -> Iterator[str]:
    prompt = build_prompt(query, docs)
    return stream_text(client=client, model=model, prompt=prompt)


def answer_structured(
    client,
    model: str,
    query: str,
    docs: List,
) -> dict:
    prompt = build_prompt(query, docs)
    function_declaration = types.FunctionDeclaration(
        name="format_answer",
        description="Return the answer and citations from the provided context.",
        parameters={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "citations"],
        },
    )
    try:
        return generate_with_function_call(
            client=client,
            model=model,
            prompt=prompt,
            function_declaration=function_declaration,
        )
    except Exception:
        response_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "citations"],
        }
        return generate_json(
            client=client,
            model=model,
            prompt=prompt,
            response_schema=response_schema,
        )
