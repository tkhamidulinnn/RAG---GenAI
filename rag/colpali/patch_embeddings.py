"""
Multimodal Patch Embeddings for ColPali-style RAG (Practice 6)

Implements Docs/ Practice 6: Multimodal RAG with ColPali-like approach.

Generates embeddings for visual patches using a Vision-Language Model (Gemini or CLIP).
These embeddings capture both textual and visual signals from document images.

Key concepts:
- Patch embeddings: Dense vectors representing visual patch content
- Query embeddings: Dense vectors for text queries (compatible with patch embeddings)
- Late interaction: Query tokens compared against all patch tokens
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from rag.colpali.visual_ingestion import PatchMetadata, PagePatches


@dataclass
class PatchEmbedding:
    """Embedding for a single patch with metadata."""
    patch_id: str
    page: int
    doc_id: str
    embedding: List[float]
    # Patch location (for visualization)
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "page": self.page,
            "doc_id": self.doc_id,
            "embedding": self.embedding,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatchEmbedding":
        return cls(**data)


def _load_patch_image(patch: PatchMetadata) -> Optional[bytes]:
    """Load patch image bytes."""
    if not patch.file_path:
        return None
    path = Path(patch.file_path)
    if not path.exists():
        return None
    return path.read_bytes()


_DESCRIPTION_PROMPT = """Describe this document patch concisely. Include:
- Any visible text (exact words)
- Visual elements (tables, charts, headers, figures)
- Layout structure
Keep it brief (1-2 sentences). If mostly blank, say "blank region"."""


def _get_patch_description_gemini(
    patch_image_bytes: bytes,
    client,
    model: str = "gemini-2.0-flash",
) -> str:
    """Get textual description of a patch (one VLM call)."""
    from google.genai import types

    try:
        mime_type = "image/png"
        parts = [types.Part.from_bytes(data=patch_image_bytes, mime_type=mime_type)]
        response = client.models.generate_content(
            model=model,
            contents=[_DESCRIPTION_PROMPT, *parts],
        )
        return response.text.strip() if response.text else "document patch"
    except Exception as e:
        return f"document patch (description failed: {str(e)[:50]})"


def generate_patch_embedding_gemini(
    patch_image_bytes: bytes,
    client,
    model: str = "gemini-2.0-flash",
) -> List[float]:
    """
    Generate embedding for a patch using Gemini's vision capabilities.

    Since Gemini doesn't provide direct image embeddings, we:
    1. Generate a textual description of the patch
    2. Embed the description using text embeddings

    This is a simplified approach. A true ColPali would use
    end-to-end vision-language embeddings.
    """
    description = _get_patch_description_gemini(patch_image_bytes, client, model)
    from google.genai import types as embed_types

    try:
        embed_response = client.models.embed_content(
            model="text-embedding-004",
            contents=[description],
            config=embed_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        if embed_response and embed_response.embeddings:
            return list(embed_response.embeddings[0].values)
    except Exception as e:
        import logging
        logging.warning("Patch embedding failed, using zero vector: %s", e)
    return [0.0] * 768


def generate_patch_embedding_clip(
    patch_image_bytes: bytes,
) -> List[float]:
    """
    Generate embedding using CLIP model as fallback.

    CLIP provides true vision embeddings that are compatible
    with text embeddings in the same space.
    """
    try:
        from PIL import Image
        import io

        # Try to use sentence-transformers CLIP
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("clip-ViT-B-32")

        # Load image
        img = Image.open(io.BytesIO(patch_image_bytes))

        # Generate embedding
        embedding = model.encode(img)
        return embedding.tolist()

    except ImportError:
        # CLIP not available, return zeros
        return [0.0] * 512
    except Exception:
        return [0.0] * 512


def generate_patch_embeddings(
    patches_list: List[PagePatches],
    client=None,
    model: str = "gemini-2.0-flash",
    use_clip: bool = False,
    batch_size: int = 5,
    embed_batch_size: int = 16,
    verbose: bool = True,
) -> List[PatchEmbedding]:
    """
    Generate embeddings for all patches.

    With Gemini: phase 1 gets a description per patch (VLM); phase 2 batches
    text embeddings (fewer API calls). Use embed_batch_size to control batch size
    for the embedding API (default 16).

    Args:
        patches_list: List of PagePatches containing patch metadata
        client: Gemini client (if using Gemini)
        model: Model for embedding generation
        use_clip: Use CLIP instead of Gemini
        batch_size: Progress logging interval (every N patches)
        embed_batch_size: Batch size for embed_content calls (Gemini only)
        verbose: Print progress

    Returns:
        List of PatchEmbedding objects
    """
    from google.genai import types as embed_types

    # Flatten (patch, page_patches for doc_id) and collect patch metadata for ordering
    flat: List[Tuple[Any, str]] = []
    for page_patches in patches_list:
        for patch in page_patches.patches:
            flat.append((patch, page_patches.doc_id))

    total_patches = len(flat)
    all_embeddings: List[PatchEmbedding] = []

    if use_clip:
        for i, (patch, doc_id) in enumerate(flat):
            patch_bytes = _load_patch_image(patch)
            if patch_bytes is None:
                embedding_values = [0.0] * 512
            else:
                embedding_values = generate_patch_embedding_clip(patch_bytes)
            all_embeddings.append(
                PatchEmbedding(
                    patch_id=patch.patch_id,
                    page=patch.page,
                    doc_id=doc_id,
                    embedding=embedding_values,
                    x=patch.x,
                    y=patch.y,
                    width=patch.width,
                    height=patch.height,
                )
            )
            if verbose and (i + 1) % batch_size == 0:
                print(f"    Embedded {i + 1}/{total_patches} patches...")
        if verbose:
            print(f"    Generated {len(all_embeddings)} patch embeddings")
        return all_embeddings

    # Gemini path: phase 1 — get description per patch (VLM)
    descriptions: List[str] = []
    patch_meta_list: List[Tuple[Any, str]] = []

    for i, (patch, doc_id) in enumerate(flat):
        patch_bytes = _load_patch_image(patch)
        if patch_bytes is None:
            descriptions.append("blank region")
        else:
            descriptions.append(_get_patch_description_gemini(patch_bytes, client, model))
        patch_meta_list.append((patch, doc_id))
        if verbose and (i + 1) % batch_size == 0:
            print(f"    Described {i + 1}/{total_patches} patches...")

    if verbose:
        print(f"    Described {len(descriptions)} patches; batch-embedding (batch_size={embed_batch_size})...")

    # Phase 2 — batch embed descriptions (fewer API calls)
    embedding_vectors: List[List[float]] = []
    for start in range(0, len(descriptions), embed_batch_size):
        batch = descriptions[start : start + embed_batch_size]
        try:
            resp = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
                config=embed_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            if resp and resp.embeddings:
                for emb in resp.embeddings:
                    embedding_vectors.append(list(emb.values))
            else:
                embedding_vectors.extend([[0.0] * 768] * len(batch))
        except Exception as e:
            import logging
            logging.warning("Batch embed failed for %d descriptions: %s", len(batch), e)
            embedding_vectors.extend([[0.0] * 768] * len(batch))

    for (patch, doc_id), embedding_values in zip(patch_meta_list, embedding_vectors):
        all_embeddings.append(
            PatchEmbedding(
                patch_id=patch.patch_id,
                page=patch.page,
                doc_id=doc_id,
                embedding=embedding_values,
                x=patch.x,
                y=patch.y,
                width=patch.width,
                height=patch.height,
            )
        )

    if verbose:
        print(f"    Generated {len(all_embeddings)} patch embeddings")

    return all_embeddings


def generate_query_embedding(
    query: str,
    client=None,
    use_clip: bool = False,
) -> List[float]:
    """
    Generate embedding for a text query.

    The query embedding must be in the same space as patch embeddings
    for late interaction comparison.

    Args:
        query: Text query
        client: Gemini client (if using Gemini)
        use_clip: Use CLIP text encoder

    Returns:
        Query embedding vector
    """
    if use_clip:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("clip-ViT-B-32")
            embedding = model.encode(query)
            return embedding.tolist()
        except ImportError:
            return [0.0] * 512

    # Use Gemini text embedding
    from google.genai import types as embed_types

    try:
        embed_response = client.models.embed_content(
            model="text-embedding-004",
            contents=[query],
            config=embed_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )

        if embed_response and embed_response.embeddings:
            return list(embed_response.embeddings[0].values)
    except Exception as e:
        import logging
        logging.warning("Query embedding failed, using zero vector: %s", e)

    return [0.0] * 768


def save_embeddings(embeddings: List[PatchEmbedding], output_dir: Path) -> None:
    """Save patch embeddings to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON for simplicity
    # In production, use a more efficient format (numpy, parquet)
    data = {
        "embeddings": [e.to_dict() for e in embeddings],
        "dimension": len(embeddings[0].embedding) if embeddings else 0,
        "count": len(embeddings),
    }

    output_file = output_dir / "patch_embeddings.json"
    output_file.write_text(json.dumps(data), encoding="utf-8")

    # Also save embeddings as numpy array for faster loading
    if embeddings:
        emb_array = np.array([e.embedding for e in embeddings], dtype=np.float32)
        np.save(str(output_dir / "embeddings.npy"), emb_array)


def load_embeddings(embeddings_dir: Path) -> Tuple[List[PatchEmbedding], Optional[np.ndarray]]:
    """
    Load patch embeddings from disk.

    Returns:
        Tuple of (list of PatchEmbedding, numpy array of embeddings)
    """
    metadata_file = embeddings_dir / "patch_embeddings.json"
    embeddings_file = embeddings_dir / "embeddings.npy"

    if not metadata_file.exists():
        return [], None

    with metadata_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = [PatchEmbedding.from_dict(e) for e in data.get("embeddings", [])]

    # Load numpy array if available
    emb_array = None
    if embeddings_file.exists():
        emb_array = np.load(str(embeddings_file))

    return embeddings, emb_array


def build_faiss_index(embeddings: List[PatchEmbedding], output_dir: Path) -> None:
    """
    Build a FAISS index for fast similarity search.

    This enables efficient retrieval but doesn't capture the full
    late interaction scoring. Used for initial candidate retrieval.
    """
    import faiss

    if not embeddings:
        return

    # Convert to numpy array
    emb_array = np.array([e.embedding for e in embeddings], dtype=np.float32)
    dimension = emb_array.shape[1]

    # Build index
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    faiss.normalize_L2(emb_array)  # Normalize for cosine similarity
    index.add(emb_array)

    # Save index
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "patch_index.faiss"))


def load_faiss_index(index_dir: Path):
    """Load FAISS index from disk."""
    import faiss

    index_file = index_dir / "patch_index.faiss"
    if not index_file.exists():
        return None

    return faiss.read_index(str(index_file))
