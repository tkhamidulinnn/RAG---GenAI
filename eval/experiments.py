"""
Experiment Runner for RAG Evaluation

Runs comparative experiments:
- FAISS vs Qdrant
- Multiple Top-K values (3, 5, 8)
- Dense vs Sparse vs Hybrid retrieval (Practice 3)
- With and without re-ranking (Practice 3)

Results are saved for analysis and comparison.
"""
from __future__ import annotations

import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Optional

from eval.runner import run_evaluation, load_evaluation_dataset
from eval.metrics import compute_metrics_batch
from eval.llm_judge import judge_batch

from rag.settings import load_settings
from rag.gemini_client import build_client


def run_experiment(
    dataset_path: Path,
    output_dir: Path,
    vector_store: str,
    top_k: int,
    retrieval_mode: str = "dense",
    enable_reranking: bool = False,
    rerank_top_n: int = 20,
    compute_metrics: bool = True,
    run_judge: bool = False,
) -> dict:
    """
    Run a single experiment configuration.

    Args:
        dataset_path: Path to evaluation CSV
        output_dir: Directory for results
        vector_store: "faiss" or "qdrant"
        top_k: Final number of documents
        retrieval_mode: "dense", "sparse", or "hybrid"
        enable_reranking: Use cross-encoder re-ranking
        rerank_top_n: Candidates before re-ranking
        compute_metrics: Compute RAGAS metrics
        run_judge: Run LLM judge

    Returns:
        Dict with experiment config and aggregated results
    """
    # Build experiment label
    mode_label = retrieval_mode.upper()
    rerank_label = "+Rerank" if enable_reranking else ""
    exp_label = f"{vector_store.upper()} | {mode_label}{rerank_label} | K={top_k}"

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_label}")
    print(f"{'='*60}")

    # Run evaluation
    samples = run_evaluation(
        dataset_path=dataset_path,
        output_dir=output_dir,
        vector_store=vector_store,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        enable_reranking=enable_reranking,
        rerank_top_n=rerank_top_n,
    )

    # Convert to dicts
    sample_dicts = [
        {
            "question": s.question,
            "generated_answer": s.generated_answer,
            "ground_truth_answer": s.ground_truth_answer,
            "ground_truth_context": s.ground_truth_context,
            "retrieved_contexts": s.retrieved_contexts,
            "retrieved_pages": s.retrieved_pages,
            "content_type": s.content_type,
            "page_number": s.page_number,
            "vector_store": s.vector_store,
            "top_k": s.top_k,
            "retrieval_mode": s.retrieval_mode,
            "reranked": s.reranked,
        }
        for s in samples
    ]

    results = {
        "config": {
            "vector_store": vector_store,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "reranked": enable_reranking,
            "num_samples": len(samples),
            "timestamp": datetime.now().isoformat(),
        },
        "samples": sample_dicts,
    }

    if compute_metrics:
        print("\nComputing RAGAS metrics...")
        settings = load_settings()
        client = build_client(settings)

        metrics_results = compute_metrics_batch(sample_dicts, client, settings.gemini_model)

        # Aggregate metrics
        agg_metrics = {
            "faithfulness": sum(r["faithfulness"] for r in metrics_results) / len(metrics_results),
            "answer_relevance": sum(r["answer_relevance"] for r in metrics_results) / len(metrics_results),
            "context_precision": sum(r["context_precision"] for r in metrics_results) / len(metrics_results),
            "context_recall": sum(r["context_recall"] for r in metrics_results) / len(metrics_results),
        }

        results["metrics_per_sample"] = [
            {
                "question": r["question"][:50],
                "faithfulness": r["faithfulness"],
                "answer_relevance": r["answer_relevance"],
                "context_precision": r["context_precision"],
                "context_recall": r["context_recall"],
            }
            for r in metrics_results
        ]
        results["aggregated_metrics"] = agg_metrics

        print(f"\nAggregated Metrics:")
        for k, v in agg_metrics.items():
            print(f"  {k}: {v:.3f}")

    if run_judge:
        print("\nRunning LLM Judge on complex samples...")
        settings = load_settings()
        client = build_client(settings)

        judged = judge_batch(sample_dicts, client, settings.gemini_model)
        judge_scores = [s.get("judge_score", None) for s in judged if s.get("judge_score") is not None]

        if judge_scores:
            results["judge_avg_score"] = sum(judge_scores) / len(judge_scores)
            print(f"  Judge Avg Score: {results['judge_avg_score']:.3f}")

    return results


def run_all_experiments(
    dataset_path: Path = Path("eval/evaluation_dataset.csv"),
    output_dir: Path = Path("eval/results"),
    stores: Optional[List[str]] = None,
    top_k_values: Optional[List[int]] = None,
    retrieval_modes: Optional[List[str]] = None,
    test_reranking: bool = False,
    compute_metrics: bool = True,
    run_judge: bool = False,
) -> List[dict]:
    """
    Run all experiment configurations.

    Args:
        dataset_path: Path to evaluation CSV
        output_dir: Directory for results
        stores: Vector stores to test (default: ["faiss"])
        top_k_values: Top-K values to test (default: [3, 5, 8])
        retrieval_modes: Retrieval modes to test (default: ["dense"])
        test_reranking: Whether to test with and without re-ranking
        compute_metrics: Whether to compute RAGAS metrics
        run_judge: Whether to run LLM judge

    Returns:
        List of experiment results
    """
    stores = stores or ["faiss"]
    top_k_values = top_k_values or [3, 5, 8]
    retrieval_modes = retrieval_modes or ["dense"]
    reranking_options = [False, True] if test_reranking else [False]

    all_results = []

    # Generate all experiment combinations
    experiments = list(product(stores, retrieval_modes, top_k_values, reranking_options))
    total = len(experiments)

    print(f"\nRunning {total} experiments...")
    print(f"Stores: {stores}")
    print(f"Modes: {retrieval_modes}")
    print(f"Top-K values: {top_k_values}")
    print(f"Re-ranking: {reranking_options}")

    for i, (store, mode, k, rerank) in enumerate(experiments, 1):
        print(f"\n[{i}/{total}] ", end="")
        try:
            result = run_experiment(
                dataset_path=dataset_path,
                output_dir=output_dir,
                vector_store=store,
                top_k=k,
                retrieval_mode=mode,
                enable_reranking=rerank,
                compute_metrics=compute_metrics,
                run_judge=run_judge,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Experiment {store}/{mode}/k={k}/rerank={rerank} failed: {e}")
            all_results.append({
                "config": {
                    "vector_store": store,
                    "top_k": k,
                    "retrieval_mode": mode,
                    "reranked": rerank,
                    "error": str(e),
                },
            })

    # Save combined results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = output_dir / f"experiments_{timestamp}.json"

    with combined_file.open("w", encoding="utf-8") as f:
        # Save without full samples to keep file manageable
        summary = [
            {
                "config": r["config"],
                "aggregated_metrics": r.get("aggregated_metrics", {}),
                "judge_avg_score": r.get("judge_avg_score"),
            }
            for r in all_results
        ]
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All experiments complete. Summary saved to {combined_file}")
    print(f"{'='*60}")

    return all_results


def generate_comparison_report(results: List[dict]) -> str:
    """Generate a markdown comparison report from experiment results."""
    lines = [
        "# RAG Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Experiment Configurations",
        "",
    ]

    # Check if we have Practice 3 features
    has_modes = any(r.get("config", {}).get("retrieval_mode", "dense") != "dense" for r in results)
    has_rerank = any(r.get("config", {}).get("reranked", False) for r in results)

    # Build appropriate header
    if has_modes or has_rerank:
        header = "| Vector Store | Mode | Reranked | Top-K | Faithfulness | Answer Relevance | Context Precision | Context Recall |"
        separator = "|--------------|------|----------|-------|--------------|------------------|-------------------|----------------|"
    else:
        header = "| Vector Store | Top-K | Faithfulness | Answer Relevance | Context Precision | Context Recall |"
        separator = "|--------------|-------|--------------|------------------|-------------------|----------------|"

    lines.append(header)
    lines.append(separator)

    for r in results:
        config = r.get("config", {})
        metrics = r.get("aggregated_metrics", {})

        if "error" in config:
            if has_modes or has_rerank:
                lines.append(
                    f"| {config.get('vector_store', 'N/A')} | "
                    f"{config.get('retrieval_mode', 'dense')} | "
                    f"{config.get('reranked', False)} | "
                    f"{config.get('top_k', 'N/A')} | "
                    f"ERROR: {config['error'][:20]}... ||||"
                )
            else:
                lines.append(
                    f"| {config.get('vector_store', 'N/A')} | {config.get('top_k', 'N/A')} | "
                    f"ERROR: {config['error'][:30]} |||"
                )
        else:
            if has_modes or has_rerank:
                lines.append(
                    f"| {config.get('vector_store', 'N/A').upper()} | "
                    f"{config.get('retrieval_mode', 'dense')} | "
                    f"{'Yes' if config.get('reranked', False) else 'No'} | "
                    f"{config.get('top_k', 'N/A')} | "
                    f"{metrics.get('faithfulness', 0):.3f} | "
                    f"{metrics.get('answer_relevance', 0):.3f} | "
                    f"{metrics.get('context_precision', 0):.3f} | "
                    f"{metrics.get('context_recall', 0):.3f} |"
                )
            else:
                lines.append(
                    f"| {config.get('vector_store', 'N/A').upper()} | {config.get('top_k', 'N/A')} | "
                    f"{metrics.get('faithfulness', 0):.3f} | "
                    f"{metrics.get('answer_relevance', 0):.3f} | "
                    f"{metrics.get('context_precision', 0):.3f} | "
                    f"{metrics.get('context_recall', 0):.3f} |"
                )

    lines.extend([
        "",
        "## Metric Definitions",
        "",
        "- **Faithfulness**: Answer grounded in retrieved context (no hallucination)",
        "- **Answer Relevance**: Answer directly addresses the question",
        "- **Context Precision**: Retrieved contexts are relevant (low noise)",
        "- **Context Recall**: Retrieved contexts cover ground truth information",
        "",
    ])

    # Add retrieval mode explanations if applicable
    if has_modes:
        lines.extend([
            "## Retrieval Modes (Practice 3)",
            "",
            "- **Dense**: Embedding-based semantic search (baseline)",
            "- **Sparse**: BM25 lexical/keyword search",
            "- **Hybrid**: Combined dense + sparse with RRF fusion",
            "",
        ])

    if has_rerank:
        lines.extend([
            "## Re-ranking",
            "",
            "Cross-encoder re-ranking retrieves more candidates (default: 20) and",
            "then re-scores them using a cross-encoder model for improved precision.",
            "",
        ])

    lines.extend([
        "## Observations",
        "",
        "_To be filled after analysis_",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation experiments")
    parser.add_argument("--dataset", type=Path, default=Path("eval/evaluation_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("eval/results"))
    parser.add_argument("--stores", nargs="+", default=["faiss"])
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--metrics", action="store_true", default=True)
    parser.add_argument("--judge", action="store_true", default=False)

    # Practice 3: Advanced retrieval options
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["dense", "sparse", "hybrid"],
        default=["dense"],
        help="Retrieval modes to test",
    )
    parser.add_argument(
        "--test-reranking",
        action="store_true",
        help="Test with and without re-ranking",
    )
    args = parser.parse_args()

    results = run_all_experiments(
        dataset_path=args.dataset,
        output_dir=args.output,
        stores=args.stores,
        top_k_values=args.top_k,
        retrieval_modes=args.modes,
        test_reranking=args.test_reranking,
        compute_metrics=args.metrics,
        run_judge=args.judge,
    )

    # Generate report
    report = generate_comparison_report(results)
    report_file = args.output / "evaluation_report.md"
    report_file.write_text(report)
    print(f"Report saved to {report_file}")
