from __future__ import annotations

from typing import Iterable, List

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise RuntimeError(
        "google-genai not found. Run: pip install -r requirements.txt"
    ) from exc
from langchain_core.embeddings import Embeddings


class GeminiEmbeddings(Embeddings):
    def __init__(
        self,
        client: genai.Client,
        model: str = "text-embedding-004",
        batch_size: int = 16,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed_texts(self, texts: Iterable[str], task_type: str) -> List[List[float]]:
        texts_list = list(texts)
        results: List[List[float]] = []
        for i in range(0, len(texts_list), self._batch_size):
            batch = texts_list[i : i + self._batch_size]
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            if not response or not response.embeddings:
                raise RuntimeError("Gemini embeddings returned empty response.")
            for embedding in response.embeddings:
                results.append(list(embedding.values))
        return results
