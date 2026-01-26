"""
LLM-as-a-Judge Evaluation

Implements LLM-based evaluation for cases where automated metrics are insufficient:
- Refusal correctness: Verify correct refusal when expected
- Nuanced evaluation: Correctness, completeness, faithfulness for complex answers

Following HuggingFace RAG Evaluation Cookbook and PromptingGuide best practices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rag.settings import load_settings
from rag.gemini_client import build_client, generate_json


@dataclass
class JudgeVerdict:
    """Verdict from LLM judge evaluation."""
    score: float  # 0.0 to 1.0
    correctness: Optional[float] = None
    completeness: Optional[float] = None
    faithfulness: Optional[float] = None
    is_refusal: bool = False
    reasoning: str = ""


def _call_judge(client, model: str, prompt: str, schema: dict) -> dict:
    """Call LLM judge with structured output."""
    try:
        return generate_json(client, model, prompt, schema)
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Judge error: {e}"}


def evaluate_refusal(
    client,
    model: str,
    question: str,
    answer: str,
    contexts: List[str],
) -> JudgeVerdict:
    """
    Evaluate if the model correctly refused to answer when it should.

    Checks:
    1. Did the model refuse (not provide a fabricated answer)?
    2. Is the refusal appropriate (information truly not in context)?
    3. Are there any hallucinated facts despite refusal?
    """
    context_text = "\n\n".join(contexts) if contexts else "(no context retrieved)"

    prompt = f"""You are a judge evaluating whether a RAG system correctly refused to answer a question.

A correct refusal means:
1. The model indicated it cannot answer (e.g., "I don't have information about...", "The context doesn't contain...")
2. The model did NOT make up facts not present in the context
3. If partial information exists, the model may provide it while noting limitations

Question:
{question}

Retrieved Context:
{context_text}

Model's Answer:
{answer}

Evaluate:
1. is_refusal: Did the model refuse or indicate inability to fully answer? (true/false)
2. score: How appropriate was the response? (0.0-1.0)
   - 1.0: Perfect refusal or honest partial answer
   - 0.5: Partial refusal but some speculation
   - 0.0: Made up facts not in context
3. has_hallucination: Did the answer contain fabricated facts? (true/false)
4. reasoning: Explain your evaluation"""

    schema = {
        "type": "object",
        "properties": {
            "is_refusal": {"type": "boolean"},
            "score": {"type": "number"},
            "has_hallucination": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["is_refusal", "score", "has_hallucination", "reasoning"],
    }

    result = _call_judge(client, model, prompt, schema)

    return JudgeVerdict(
        score=min(1.0, max(0.0, float(result.get("score", 0.0)))),
        is_refusal=result.get("is_refusal", False),
        reasoning=result.get("reasoning", ""),
    )


def evaluate_nuanced_answer(
    client,
    model: str,
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
) -> JudgeVerdict:
    """
    Evaluate nuanced or explanatory answers on multiple dimensions.

    Used for questions requiring:
    - Summarization
    - Comparison
    - Multi-step reasoning
    - Explanation

    Evaluates:
    - Correctness: Are the facts accurate?
    - Completeness: Does it cover all key points?
    - Faithfulness: Is it grounded in context?
    """
    context_text = "\n\n".join(contexts) if contexts else "(no context)"

    prompt = f"""You are a judge evaluating the quality of a RAG system's answer to a complex question.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context_text}

Model's Answer:
{answer}

Evaluate the model's answer on three dimensions:

1. correctness (0.0-1.0): Are the facts in the answer accurate?
   - Compare against ground truth and context
   - Penalize factual errors

2. completeness (0.0-1.0): Does the answer cover the key points?
   - Compare against ground truth for coverage
   - Partial answers get partial scores

3. faithfulness (0.0-1.0): Is the answer grounded in the retrieved context?
   - Information should come from context, not fabricated
   - Allow reasonable inferences

4. overall_score (0.0-1.0): Weighted average considering all factors

5. reasoning: Explain your evaluation"""

    schema = {
        "type": "object",
        "properties": {
            "correctness": {"type": "number"},
            "completeness": {"type": "number"},
            "faithfulness": {"type": "number"},
            "overall_score": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["correctness", "completeness", "faithfulness", "overall_score", "reasoning"],
    }

    result = _call_judge(client, model, prompt, schema)

    return JudgeVerdict(
        score=min(1.0, max(0.0, float(result.get("overall_score", 0.0)))),
        correctness=min(1.0, max(0.0, float(result.get("correctness", 0.0)))),
        completeness=min(1.0, max(0.0, float(result.get("completeness", 0.0)))),
        faithfulness=min(1.0, max(0.0, float(result.get("faithfulness", 0.0)))),
        reasoning=result.get("reasoning", ""),
    )


def judge_sample(
    sample: dict,
    client=None,
    model: str = None,
    is_refusal_expected: bool = False,
) -> dict:
    """
    Apply LLM judge to a single evaluation sample.

    Args:
        sample: Dict with question, generated_answer, retrieved_contexts, ground_truth_answer
        client: Gemini client
        model: Model name
        is_refusal_expected: Whether refusal is the correct behavior

    Returns:
        Dict with judge verdict fields added
    """
    if client is None:
        settings = load_settings()
        client = build_client(settings)
        model = model or settings.gemini_model

    question = sample["question"]
    answer = sample["generated_answer"]
    contexts = sample["retrieved_contexts"]
    ground_truth = sample.get("ground_truth_answer", "")

    if is_refusal_expected:
        verdict = evaluate_refusal(client, model, question, answer, contexts)
        return {
            **sample,
            "judge_score": verdict.score,
            "judge_is_refusal": verdict.is_refusal,
            "judge_reasoning": verdict.reasoning,
        }
    else:
        verdict = evaluate_nuanced_answer(
            client, model, question, answer, ground_truth, contexts
        )
        return {
            **sample,
            "judge_score": verdict.score,
            "judge_correctness": verdict.correctness,
            "judge_completeness": verdict.completeness,
            "judge_faithfulness": verdict.faithfulness,
            "judge_reasoning": verdict.reasoning,
        }


def judge_batch(
    samples: List[dict],
    client=None,
    model: str = None,
    nuanced_indices: Optional[List[int]] = None,
) -> List[dict]:
    """
    Apply LLM judge to a batch of samples.

    Args:
        samples: List of evaluation samples
        client: Gemini client
        model: Model name
        nuanced_indices: Indices of samples requiring nuanced evaluation (others skipped)

    Returns:
        List of samples with judge verdicts added
    """
    if client is None:
        settings = load_settings()
        client = build_client(settings)
        model = model or settings.gemini_model

    if nuanced_indices is None:
        # By default, evaluate samples with complex content types
        nuanced_indices = [
            i for i, s in enumerate(samples)
            if s.get("content_type", "") in ("combination", "image", "combination (table and text)")
        ]

    results = []
    for i, sample in enumerate(samples):
        if i in nuanced_indices:
            print(f"[Judge {i+1}/{len(samples)}] Evaluating: {sample['question'][:50]}...")
            result = judge_sample(sample, client, model, is_refusal_expected=False)
        else:
            result = {**sample}  # Skip judge for straightforward samples
        results.append(result)

    return results
