from __future__ import annotations

from typing import Any, Dict, Optional

from rag.settings import Settings


def build_langfuse(settings: Settings):
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse import Langfuse
    except Exception:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def start_trace(langfuse, name: str, input_payload: Dict[str, Any]):
    if not langfuse:
        return None
    return langfuse.trace(name=name, input=input_payload)


def end_trace(trace, output_payload: Dict[str, Any]):
    if not trace:
        return
    trace.update(output=output_payload)
