from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from rag.settings import load_settings
from rag.gemini_client import build_client
from rag.embeddings import GeminiEmbeddings
from rag.indexing import (
    build_documents,
    split_documents,
    build_faiss,
    build_qdrant,
    build_bm25_index,
)
from rag.pdf_extract import extract_all

# Practice 5: Multimodal imports
from rag.table_extraction import extract_tables, save_tables
from rag.image_extraction import extract_images_from_pdf, save_images
from rag.multimodal_indexing import (
    tables_to_documents,
    images_to_documents,
    build_table_index,
    build_image_index,
)


def read_text_pages(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="Extract PDF content and build vectorstores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic ingestion (text-only, FAISS + Qdrant)
    python ingest.py

    # Quick test with FAISS only
    python ingest.py --skip-qdrant

    # With advanced retrieval features (Practice 3)
    python ingest.py --extract-metadata --build-bm25

    # Full ingestion with all features (Practice 3)
    python ingest.py --extract-metadata --build-bm25

    # Multimodal ingestion (Practice 5)
    python ingest.py --multimodal --skip-qdrant

    # Full multimodal with all features
    python ingest.py --extract-metadata --build-bm25 --multimodal --skip-qdrant
        """,
    )
    parser.add_argument("--pdf", type=Path, default=settings.pdf_path, help="Path to input PDF")
    parser.add_argument("--artifacts", type=Path, default=settings.artifacts_dir, help="Artifacts output dir")
    parser.add_argument("--faiss-out", type=Path, default=settings.faiss_dir, help="FAISS output dir")
    parser.add_argument("--include-images", action="store_true", help="[OPTIONAL] Include image extraction (multimodal)")
    parser.add_argument("--include-tables", action="store_true", help="[OPTIONAL] Include table extraction")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant indexing")

    # Practice 3: Advanced retrieval options
    parser.add_argument(
        "--extract-metadata",
        action="store_true",
        help="[Practice 3] Extract section and content type metadata for filtering",
    )
    parser.add_argument(
        "--build-bm25",
        action="store_true",
        help="[Practice 3] Build BM25 index for sparse/hybrid retrieval",
    )

    # Practice 5: Multimodal options
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="[Practice 5] Enable multimodal extraction and indexing (tables + images)",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="[Practice 5] Extract and index tables only (no images)",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="[Practice 5] Extract and index images only (no tables)",
    )
    args = parser.parse_args()

    pdf_path = args.pdf.expanduser().resolve()
    artifacts_dir = args.artifacts.expanduser().resolve()
    faiss_dir = args.faiss_out.expanduser().resolve()
    bm25_path = faiss_dir.parent / "bm25_index.json"

    # Practice 5: Multimodal paths
    tables_dir = artifacts_dir / "tables_extracted"
    images_dir = artifacts_dir / "images_extracted"
    table_index_dir = faiss_dir.parent / "table_index"
    image_index_dir = faiss_dir.parent / "image_index"

    # Determine multimodal settings
    extract_tables_flag = args.multimodal or args.tables_only
    extract_images_flag = args.multimodal or args.images_only

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    client = build_client(settings)

    # Step 1: Extract text from PDF
    print("Step 1: Extracting PDF content...")
    extraction_summary = extract_all(
        pdf_path=pdf_path,
        artifacts_dir=artifacts_dir,
        gemini_model=settings.gemini_model,
        client=client,
        skip_images=not args.include_images,
        skip_tables=not args.include_tables,
    )
    print(f"  Extracted: {extraction_summary['text_pages']} pages")

    # Step 2: Build documents with optional metadata extraction
    print("Step 2: Building documents...")
    text_pages_path = artifacts_dir / "text" / "pages.jsonl"
    pages = read_text_pages(text_pages_path)
    documents = build_documents(pages, extract_metadata=args.extract_metadata)
    if args.extract_metadata:
        print(f"  Metadata extracted for {len(documents)} documents")

    # Step 3: Split into chunks with optional metadata enrichment
    print("Step 3: Chunking documents...")
    chunks = split_documents(documents, enrich_metadata=args.extract_metadata)
    print(f"  Created {len(chunks)} chunks")

    # Step 4: Build embeddings
    print("Step 4: Building embeddings and vector stores...")
    embeddings = GeminiEmbeddings(client=client, model=settings.embedding_model)

    # Build FAISS
    build_faiss(chunks, embeddings, faiss_dir)
    print(f"  FAISS index saved to {faiss_dir}")

    # Build Qdrant (optional)
    if not args.skip_qdrant:
        build_qdrant(chunks, embeddings, settings.qdrant_path, settings.qdrant_collection)
        print(f"  Qdrant collection saved to {settings.qdrant_path}")

    # Step 5: Build BM25 index for sparse retrieval (Practice 3)
    if args.build_bm25:
        print("Step 5: Building BM25 index for sparse retrieval...")
        build_bm25_index(chunks, bm25_path)
        print(f"  BM25 index saved to {bm25_path}")

    # Step 6: Practice 5 - Multimodal extraction and indexing
    tables_count = 0
    images_count = 0

    if extract_tables_flag:
        print("\nStep 6a: Extracting tables (Practice 5)...")
        try:
            extracted_tables = extract_tables(pdf_path, doc_id="ifc_2024")
            tables_count = len(extracted_tables)
            print(f"  Found {tables_count} tables")

            if extracted_tables:
                save_tables(extracted_tables, tables_dir)
                print(f"  Tables saved to {tables_dir}")

                # Build table index
                print("  Building table index...")
                table_docs = tables_to_documents(extracted_tables)
                if table_docs:
                    build_table_index(extracted_tables, embeddings, table_index_dir)
                    print(f"  Table index saved to {table_index_dir}")
        except Exception as e:
            print(f"  Warning: Table extraction failed: {e}")

    if extract_images_flag:
        print("\nStep 6b: Extracting images/charts (Practice 5)...")
        try:
            extracted_images = extract_images_from_pdf(
                pdf_path=pdf_path,
                output_dir=images_dir,
                doc_id="ifc_2024",
                client=client,
                model=settings.gemini_model,
                use_vlm=True,
            )
            images_count = len(extracted_images)
            print(f"  Found {images_count} images/charts")

            if extracted_images:
                save_images(extracted_images, images_dir)
                print(f"  Images saved to {images_dir}")

                # Build image index
                print("  Building image index...")
                image_docs = images_to_documents(extracted_images)
                if image_docs:
                    build_image_index(extracted_images, embeddings, image_index_dir)
                    print(f"  Image index saved to {image_index_dir}")
        except Exception as e:
            print(f"  Warning: Image extraction failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Pages extracted:    {extraction_summary['text_pages']}")
    print(f"  Chunks created:     {len(chunks)}")
    print(f"  FAISS index:        {faiss_dir}")
    print(f"  Qdrant collection:  {'skipped' if args.skip_qdrant else settings.qdrant_collection}")
    print(f"  BM25 index:         {'skipped' if not args.build_bm25 else bm25_path}")
    print(f"  Metadata extracted: {'yes' if args.extract_metadata else 'no'}")
    if extract_tables_flag or extract_images_flag:
        print(f"  --- Practice 5 (Multimodal) ---")
        print(f"  Tables extracted:   {tables_count}")
        print(f"  Images extracted:   {images_count}")
        print(f"  Table index:        {table_index_dir if tables_count > 0 else 'none'}")
        print(f"  Image index:        {image_index_dir if images_count > 0 else 'none'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
