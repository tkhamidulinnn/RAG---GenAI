from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from rag.settings import load_settings
from rag.gemini_client import build_client
from rag.embeddings import GeminiEmbeddings
from rag.indexing import build_documents, split_documents, build_faiss, build_qdrant
from rag.pdf_extract import extract_all


def read_text_pages(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Extract PDF content and build vectorstores.")
    parser.add_argument("--pdf", type=Path, default=settings.pdf_path, help="Path to input PDF")
    parser.add_argument("--artifacts", type=Path, default=settings.artifacts_dir, help="Artifacts output dir")
    parser.add_argument("--faiss-out", type=Path, default=settings.faiss_dir, help="FAISS output dir")
    parser.add_argument("--include-images", action="store_true", help="[OPTIONAL] Include image extraction (multimodal)")
    parser.add_argument("--include-tables", action="store_true", help="[OPTIONAL] Include table extraction")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant indexing")
    args = parser.parse_args()

    pdf_path = args.pdf.expanduser().resolve()
    artifacts_dir = args.artifacts.expanduser().resolve()
    faiss_dir = args.faiss_out.expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    client = build_client(settings)

    extraction_summary = extract_all(
        pdf_path=pdf_path,
        artifacts_dir=artifacts_dir,
        gemini_model=settings.gemini_model,
        client=client,
        skip_images=not args.include_images,
        skip_tables=not args.include_tables,
    )

    text_pages_path = artifacts_dir / "text" / "pages.jsonl"
    pages = read_text_pages(text_pages_path)
    documents = build_documents(pages)
    chunks = split_documents(documents)

    embeddings = GeminiEmbeddings(client=client, model=settings.embedding_model)
    build_faiss(chunks, embeddings, faiss_dir)

    if not args.skip_qdrant:
        build_qdrant(chunks, embeddings, settings.qdrant_path, settings.qdrant_collection)

    print(
        "Ingest done: "
        f"pages={extraction_summary['text_pages']}, "
        f"chunks={len(chunks)}, "
        f"faiss={faiss_dir}, "
        f"qdrant={'skipped' if args.skip_qdrant else settings.qdrant_collection}"
    )


if __name__ == "__main__":
    main()
