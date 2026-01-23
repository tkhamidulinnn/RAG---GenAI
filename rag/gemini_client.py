from __future__ import annotations

from typing import Iterable, Iterator
import json

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise RuntimeError(
        "google-genai not found. Run: pip install -r requirements.txt"
    ) from exc

from rag.settings import Settings


def build_client(settings: Settings) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=settings.gcp_project,
        location=settings.gcp_location,
    )


def generate_json(
    client: genai.Client,
    model: str,
    prompt: str,
    response_schema: dict,
    parts: Iterable[types.Part] | None = None,
) -> dict:
    contents = [prompt]
    if parts:
        contents.extend(parts)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )
    if not response or not response.text:
        raise RuntimeError("Gemini returned empty structured response.")
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {response.text}") from exc


def stream_text(
    client: genai.Client,
    model: str,
    prompt: str,
    parts: Iterable[types.Part] | None = None,
) -> Iterator[str]:
    contents = [prompt]
    if parts:
        contents.extend(parts)

    stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def generate_with_function_call(
    client: genai.Client,
    model: str,
    prompt: str,
    function_declaration: types.FunctionDeclaration,
) -> dict:
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[function_declaration])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        ),
    )
    if not response.candidates:
        raise RuntimeError("Gemini returned no candidates for function call.")
    parts = response.candidates[0].content.parts
    for part in parts:
        if part.function_call:
            return part.function_call.args or {}
    raise RuntimeError("Gemini did not produce a function call.")
