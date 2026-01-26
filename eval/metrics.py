"""
RAGAS-Style Metrics for RAG Evaluation

Implements the core RAGAS metrics:
- Faithfulness: Is the answer grounded in retrieved context?
- Answer Relevance: Is the answer relevant to the question?
- Context Precision: Are retrieved contexts relevant (low noise)?
- Context Recall: Do retrieved contexts cover the ground truth?

These metrics use LLM-based evaluation following RAGAS methodology.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from rag.settings import load_settings
from rag.gemini_client import build_client, generate_json


@dataclass
class MetricScores:
    """Scores for a single evaluation sample."""
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


def _call_llm_judge(client, model: str, prompt: str) -> dict:
    """Call LLM to evaluate and return structured response."""
    response_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["score", "reasoning"],
    }
    try:
        return generate_json(client, model, prompt, response_schema)
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Error: {e}"}


def compute_faithfulness(
    client,
    model: str,
    answer: str,
    contexts: List[str],
) -> float:
    """
    Faithfulness: Measures if the answer is grounded in the retrieved context.

    A faithful answer only contains information that can be derived from the context.
    Score: 0.0 (hallucinated) to 1.0 (fully grounded)
    """
    context_text = "\n\n".join(contexts)

    prompt = f"""You are evaluating the faithfulness of a generated answer.

Faithfulness measures whether ALL claims in the answer can be inferred from the provided context.
A faithful answer does NOT contain information beyond what is in the context.

Context:
{context_text}

Answer to evaluate:
{answer}

Scoring guidelines:
- 1.0: All claims in the answer are directly supported by the context
- 0.7-0.9: Most claims are supported, minor additions that don't mislead
- 0.4-0.6: Some claims are supported, some are not in the context
- 0.1-0.3: Few claims are supported, significant information not in context
- 0.0: Answer contains hallucinated facts contradicting or not in context

Provide a score between 0.0 and 1.0, and explain your reasoning."""

    result = _call_llm_judge(client, model, prompt)
    return min(1.0, max(0.0, float(result.get("score", 0.0))))


def compute_answer_relevance(
    client,
    model: str,
    question: str,
    answer: str,
) -> float:
    """
    Answer Relevance: Measures if the answer addresses the question.

    A relevant answer directly responds to what was asked.
    Score: 0.0 (irrelevant) to 1.0 (fully relevant)
    """
    prompt = f"""You are evaluating the relevance of an answer to a question.

Answer relevance measures whether the answer directly addresses the question asked.
An irrelevant answer might be factually correct but not answer the question.

Question:
{question}

Answer:
{answer}

Scoring guidelines:
- 1.0: Answer directly and completely addresses the question
- 0.7-0.9: Answer mostly addresses the question with minor tangents
- 0.4-0.6: Answer partially addresses the question
- 0.1-0.3: Answer barely relates to the question
- 0.0: Answer does not address the question at all

Provide a score between 0.0 and 1.0, and explain your reasoning."""

    result = _call_llm_judge(client, model, prompt)
    return min(1.0, max(0.0, float(result.get("score", 0.0))))


def compute_context_precision(
    client,
    model: str,
    question: str,
    contexts: List[str],
) -> float:
    """
    Context Precision: Measures if retrieved contexts are relevant to the question.

    High precision means low noise (few irrelevant chunks retrieved).
    Score: 0.0 (all noise) to 1.0 (all relevant)
    """
    if not contexts:
        return 0.0

    context_text = "\n\n---\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)])

    prompt = f"""You are evaluating the precision of retrieved contexts for a question.

Context precision measures what proportion of retrieved contexts are actually relevant to answering the question.
Irrelevant contexts are noise that could mislead the answer generation.

Question:
{question}

Retrieved Contexts:
{context_text}

For each context, determine if it contains information relevant to answering the question.
Then calculate: (number of relevant contexts) / (total contexts)

Provide a score between 0.0 and 1.0, and explain which contexts are relevant/irrelevant."""

    result = _call_llm_judge(client, model, prompt)
    return min(1.0, max(0.0, float(result.get("score", 0.0))))


def compute_context_recall(
    client,
    model: str,
    ground_truth: str,
    contexts: List[str],
) -> float:
    """
    Context Recall: Measures if retrieved contexts contain the ground truth information.

    High recall means the retrieval found the necessary information.
    Score: 0.0 (ground truth not found) to 1.0 (fully covered)
    """
    if not contexts:
        return 0.0

    context_text = "\n\n".join(contexts)

    prompt = f"""You are evaluating the recall of retrieved contexts.

Context recall measures whether the retrieved contexts contain the information needed to answer correctly.
Compare the ground truth answer with the retrieved contexts.

Ground Truth Answer:
{ground_truth}

Retrieved Contexts:
{context_text}

Scoring guidelines:
- 1.0: All information in the ground truth is present in the contexts
- 0.7-0.9: Most key information is present
- 0.4-0.6: Some key information is present
- 0.1-0.3: Little key information is present
- 0.0: None of the ground truth information is in the contexts

Provide a score between 0.0 and 1.0, and explain your reasoning."""

    result = _call_llm_judge(client, model, prompt)
    return min(1.0, max(0.0, float(result.get("score", 0.0))))


def compute_all_metrics(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    client=None,
    model: str = None,
) -> MetricScores:
    """
    Compute all RAGAS-style metrics for a single sample.

    Args:
        question: The input question
        answer: The generated answer
        contexts: List of retrieved context strings
        ground_truth: The ground truth answer
        client: Gemini client (created if None)
        model: Model name (from settings if None)

    Returns:
        MetricScores with all four metrics
    """
    if client is None:
        settings = load_settings()
        client = build_client(settings)
        model = model or settings.gemini_model

    faithfulness = compute_faithfulness(client, model, answer, contexts)
    answer_relevance = compute_answer_relevance(client, model, question, answer)
    context_precision = compute_context_precision(client, model, question, contexts)
    context_recall = compute_context_recall(client, model, ground_truth, contexts)

    return MetricScores(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        context_recall=context_recall,
    )


def compute_metrics_batch(samples: List[dict], client=None, model: str = None) -> List[dict]:
    """
    Compute metrics for a batch of evaluation samples.

    Args:
        samples: List of sample dicts with question, generated_answer, retrieved_contexts, ground_truth_answer
        client: Gemini client
        model: Model name

    Returns:
        List of dicts with original sample data plus metric scores
    """
    if client is None:
        settings = load_settings()
        client = build_client(settings)
        model = model or settings.gemini_model

    results = []
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Computing metrics for: {sample['question'][:50]}...")

        scores = compute_all_metrics(
            question=sample["question"],
            answer=sample["generated_answer"],
            contexts=sample["retrieved_contexts"],
            ground_truth=sample["ground_truth_answer"],
            client=client,
            model=model,
        )

        result = {**sample, **scores.to_dict()}
        results.append(result)

    return results
