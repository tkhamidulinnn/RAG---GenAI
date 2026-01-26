#!/usr/bin/env python3
"""
RAG Evaluation Pipeline

Single command to run the complete evaluation:
    python run_evaluation.py

Options:
    --quick     Run with single Top-K (5) for faster testing
    --full      Run all configurations including LLM judge
    --store     Specify vector store (faiss/qdrant)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_evaluation.py              # Standard evaluation (FAISS, k=3,5,8)
    python run_evaluation.py --quick      # Quick test (FAISS, k=5 only)
    python run_evaluation.py --full       # Full evaluation with LLM judge
    python run_evaluation.py --store qdrant  # Use Qdrant instead of FAISS
        """,
    )
    parser.add_argument("--quick", action="store_true", help="Quick run with k=5 only")
    parser.add_argument("--full", action="store_true", help="Full run including LLM judge")
    parser.add_argument("--store", choices=["faiss", "qdrant"], default="faiss")
    parser.add_argument("--dataset", type=Path, default=Path("eval/evaluation_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("eval/results"))
    args = parser.parse_args()

    # Check dataset exists
    if not args.dataset.exists():
        print(f"Error: Evaluation dataset not found: {args.dataset}")
        print("Expected CSV with columns: Question, Ground_Truth_Context, Ground_Truth_Answer, Page_Number, Context_Content_Type")
        sys.exit(1)

    # Import here to allow --help without GCP setup
    from eval.experiments import run_all_experiments, generate_comparison_report

    # Configure experiments
    if args.quick:
        top_k_values = [5]
        run_judge = False
        print("Running QUICK evaluation (k=5 only, no LLM judge)")
    elif args.full:
        top_k_values = [3, 5, 8]
        run_judge = True
        print("Running FULL evaluation (k=3,5,8 with LLM judge)")
    else:
        top_k_values = [3, 5, 8]
        run_judge = False
        print("Running STANDARD evaluation (k=3,5,8, no LLM judge)")

    print(f"Vector Store: {args.store.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print()

    # Run experiments
    results = run_all_experiments(
        dataset_path=args.dataset,
        output_dir=args.output,
        stores=[args.store],
        top_k_values=top_k_values,
        compute_metrics=True,
        run_judge=run_judge,
    )

    # Generate report
    report = generate_comparison_report(results)
    report_file = args.output / "evaluation_report.md"
    report_file.write_text(report)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for r in results:
        config = r.get("config", {})
        metrics = r.get("aggregated_metrics", {})

        if "error" in config:
            print(f"\n{config['vector_store']}/k={config['top_k']}: ERROR - {config['error']}")
        else:
            print(f"\n{config['vector_store'].upper()}/k={config['top_k']}:")
            print(f"  Faithfulness:       {metrics.get('faithfulness', 0):.3f}")
            print(f"  Answer Relevance:   {metrics.get('answer_relevance', 0):.3f}")
            print(f"  Context Precision:  {metrics.get('context_precision', 0):.3f}")
            print(f"  Context Recall:     {metrics.get('context_recall', 0):.3f}")

            if r.get("judge_avg_score") is not None:
                print(f"  Judge Avg Score:    {r['judge_avg_score']:.3f}")

    print("\n" + "=" * 60)
    print(f"Full report: {report_file}")
    print(f"Raw results: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
