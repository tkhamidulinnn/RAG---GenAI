"""
RAG Evaluation Runner

Executes the RAG pipeline over the evaluation dataset and collects samples
for metric computation. Ensures reproducibility by capturing all intermediate
outputs (retrieved contexts, generated answers).

Supports both baseline (Practice 1) and advanced (Practice 3) retrieval modes.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rag.settings import load_settings
from rag.gemini_client import build_client
from rag.embeddings import GeminiEmbeddings
from rag.rag_pipeline import retrieve, build_prompt

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore


@dataclass
class EvalSample:
    """Single evaluation sample with all data needed for metrics."""
    question: str
    ground_truth_answer: str
    ground_truth_context: str
    generated_answer: str
    retrieved_contexts: List[str]
    retrieved_pages: List[int]
    vector_store: str
    top_k: int
    content_type: str
    page_number: str

    # Practice 3: Advanced retrieval metadata
    retrieval_mode: str = "dense"
    reranked: bool = False
    retrieval_scores: Optional[List[float]] = None


def load_evaluation_dataset(path: Path) -> List[dict]:
    """Load evaluation questions from CSV."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question": row.get("Question", "").strip(),
                "ground_truth_context": row.get("Ground_Truth_Context", "").strip(),
                "ground_truth_answer": row.get("Ground_Truth_Answer", "").strip(),
                "page_number": row.get("Page_Number", "").strip(),
                "content_type": row.get("Context_Content_Type", "").strip(),
            })
    return rows


def load_vector_store(store_type: str, settings, client) -> FAISS | QdrantVectorStore:
    """Load the specified vector store."""
    embeddings = GeminiEmbeddings(client=client, model=settings.embedding_model)

    if store_type == "faiss":
        if not settings.faiss_dir.exists():
            raise FileNotFoundError(f"FAISS index not found: {settings.faiss_dir}")
        return FAISS.load_local(
            str(settings.faiss_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    elif store_type == "qdrant":
        if settings.qdrant_url:
            return QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                url=settings.qdrant_url,
                collection_name=settings.qdrant_collection,
            )
        if not settings.qdrant_path.exists():
            raise FileNotFoundError(f"Qdrant store not found: {settings.qdrant_path}")
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            path=str(settings.qdrant_path),
            collection_name=settings.qdrant_collection,
        )
    else:
        raise ValueError(f"Unknown vector store: {store_type}")


def load_advanced_retriever(store_type: str, settings, client):
    """Load advanced retriever with BM25 support."""
    from rag.advanced_retriever import create_advanced_retriever
    from rag.sparse_retrieval import load_bm25_index

    dense = load_vector_store(store_type, settings, client)

    # Try to load BM25 index
    bm25_path = settings.faiss_dir.parent / "bm25_index.json"
    sparse = None
    if bm25_path.exists():
        sparse = load_bm25_index(bm25_path)

    return create_advanced_retriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )


def generate_answer(client, model: str, question: str, docs: List) -> str:
    """Generate answer using the RAG pipeline."""
    from rag.gemini_client import stream_text

    prompt = build_prompt(question, docs)
    tokens = []
    for token in stream_text(client=client, model=model, prompt=prompt):
        tokens.append(token)
    return "".join(tokens)


def run_evaluation(
    dataset_path: Path,
    output_dir: Path,
    vector_store: str = "faiss",
    top_k: int = 5,
    retrieval_mode: str = "dense",
    enable_reranking: bool = False,
    rerank_top_n: int = 20,
) -> List[EvalSample]:
    """
    Run RAG evaluation over the dataset.

    Args:
        dataset_path: Path to evaluation CSV
        output_dir: Directory to save results
        vector_store: "faiss" or "qdrant"
        top_k: Number of documents to retrieve
        retrieval_mode: "dense", "sparse", or "hybrid" (Practice 3)
        enable_reranking: Whether to use cross-encoder re-ranking (Practice 3)
        rerank_top_n: Candidates to retrieve before re-ranking

    Returns:
        List of EvalSample with all evaluation data
    """
    settings = load_settings()
    client = build_client(settings)

    # Load dataset
    questions = load_evaluation_dataset(dataset_path)
    print(f"Loaded {len(questions)} evaluation questions")

    # Determine if we need advanced retriever
    use_advanced = retrieval_mode != "dense" or enable_reranking

    if use_advanced:
        from rag.retrieval_config import RetrievalConfig, RerankerConfig, HybridConfig

        retriever = load_advanced_retriever(vector_store, settings, client)
        print(f"Loaded advanced retriever (mode={retrieval_mode}, reranking={enable_reranking})")

        config = RetrievalConfig(
            mode=retrieval_mode,
            top_k=top_k,
            reranker=RerankerConfig(
                enabled=enable_reranking,
                top_n=rerank_top_n,
                top_k=top_k,
            ),
            hybrid=HybridConfig(enabled=(retrieval_mode == "hybrid")),
            preserve_scores=True,
        )
    else:
        db = load_vector_store(vector_store, settings, client)
        print(f"Loaded {vector_store.upper()} vector store (baseline mode)")
        retriever = None
        config = None

    samples: List[EvalSample] = []

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Evaluating: {q['question'][:50]}...")

        # Retrieve documents
        if use_advanced:
            result = retriever.retrieve(q["question"], config)
            docs = result.documents
            scores = result.scores
        else:
            docs = retrieve(db, q["question"], k=top_k)
            scores = None

        # Extract contexts and pages
        retrieved_contexts = [doc.page_content for doc in docs]
        retrieved_pages = [doc.metadata.get("page", 0) for doc in docs]

        # Generate answer
        answer = generate_answer(client, settings.gemini_model, q["question"], docs)

        sample = EvalSample(
            question=q["question"],
            ground_truth_answer=q["ground_truth_answer"],
            ground_truth_context=q["ground_truth_context"],
            generated_answer=answer,
            retrieved_contexts=retrieved_contexts,
            retrieved_pages=retrieved_pages,
            vector_store=vector_store,
            top_k=top_k,
            content_type=q["content_type"],
            page_number=q["page_number"],
            retrieval_mode=retrieval_mode,
            reranked=enable_reranking,
            retrieval_scores=scores,
        )
        samples.append(sample)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Include retrieval mode in filename
    mode_suffix = f"_{retrieval_mode}" if retrieval_mode != "dense" else ""
    rerank_suffix = "_reranked" if enable_reranking else ""
    output_file = output_dir / f"eval_samples_{vector_store}_k{top_k}{mode_suffix}{rerank_suffix}_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in samples], f, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} samples to {output_file}")
    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--dataset", type=Path, default=Path("eval/evaluation_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("eval/results"))
    parser.add_argument("--store", choices=["faiss", "qdrant"], default="faiss")
    parser.add_argument("--top-k", type=int, default=5)

    # Practice 3: Advanced retrieval options
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid"],
        default="dense",
        help="Retrieval mode (dense=baseline, sparse=BM25, hybrid=both)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder re-ranking",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=20,
        help="Candidates to retrieve before re-ranking",
    )
    args = parser.parse_args()

    samples = run_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output,
        vector_store=args.store,
        top_k=args.top_k,
        retrieval_mode=args.mode,
        enable_reranking=args.rerank,
        rerank_top_n=args.rerank_top_n,
    )
    print(f"Evaluation complete: {len(samples)} samples")
