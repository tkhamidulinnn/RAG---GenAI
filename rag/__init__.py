"""
RAG Practice - Foundational to Advanced RAG System

This package implements a complete RAG pipeline:
- Ingestion: PDF extraction, chunking, embedding
- Indexing: FAISS and Qdrant vector stores
- Retrieval: Dense, sparse (BM25), hybrid, with re-ranking
- Generation: Gemini-based answer generation
- Evaluation: RAGAS metrics and LLM judge

Main entry points:
- scripts/ingest.py: Build indexes from PDFs
- apps/app.py: Streamlit UI for text-based RAG
- apps/app_multimodal.py: Streamlit UI for multimodal RAG
- scripts/run_evaluation.py: Run evaluation pipeline
"""

__all__ = []
