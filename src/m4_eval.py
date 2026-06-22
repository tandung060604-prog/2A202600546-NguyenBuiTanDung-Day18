from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
    fallback = {name: 0.0 for name in metric_names}
    fallback["per_question"] = []

    lengths = {
        len(questions),
        len(answers),
        len(contexts),
        len(ground_truths),
    }
    if len(lengths) != 1:
        print("  ⚠️  RAGAS evaluation failed: input lists must have equal length")
        return fallback
    if not questions:
        return fallback

    def safe_float(value) -> float:
        try:
            number = float(value)
            return number if math.isfinite(number) else 0.0
        except (TypeError, ValueError):
            return 0.0

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        normalized_contexts = [
            [str(context) for context in context_list if context is not None]
            for context_list in contexts
        ]
        dataset = Dataset.from_dict({
            "question": [str(question) for question in questions],
            "answer": [str(answer) for answer in answers],
            "contexts": normalized_contexts,
            "ground_truth": [str(truth) for truth in ground_truths],
        })
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        dataframe = result.to_pandas()

        per_question = []
        for index, row in dataframe.iterrows():
            # Some RAGAS versions omit original input columns from the frame,
            # so the source lists remain the reliable fallback.
            source_index = int(index) if isinstance(index, int) else len(per_question)
            row_contexts = row.get("contexts", normalized_contexts[source_index])
            if not isinstance(row_contexts, list):
                row_contexts = list(row_contexts) if row_contexts is not None else []
            per_question.append(EvalResult(
                question=str(row.get("question", questions[source_index])),
                answer=str(row.get("answer", answers[source_index])),
                contexts=[str(context) for context in row_contexts],
                ground_truth=str(row.get("ground_truth", ground_truths[source_index])),
                faithfulness=safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=safe_float(row.get("context_precision", 0.0)),
                context_recall=safe_float(row.get("context_recall", 0.0)),
            ))

        aggregates = {}
        for metric_name in metric_names:
            values = [getattr(item, metric_name) for item in per_question]
            aggregates[metric_name] = (
                float(sum(values) / len(values)) if values else 0.0
            )
        aggregates["per_question"] = per_question
        return aggregates
    except Exception as error:
        print(f"  ⚠️  RAGAS evaluation failed: {error}")
        return fallback


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0 or not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "Generation failure: the answer contains claims unsupported by the retrieved context.",
            "Tighten the grounded-answer prompt, lower temperature, and require citations or refusal when evidence is missing.",
        ),
        "answer_relevancy": (
            "Generation failure: the answer does not directly address the user's question.",
            "Improve the answer prompt, preserve the original query intent, and request a concise direct answer.",
        ),
        "context_precision": (
            "Retrieval/reranking failure: too many irrelevant chunks appear above useful evidence.",
            "Improve cross-encoder reranking, metadata/version filters, and reduce the final context count.",
        ),
        "context_recall": (
            "Retrieval/chunking failure: the selected context is missing required evidence.",
            "Improve chunk boundaries, hybrid retrieval, top-k coverage, or parent-context expansion.",
        ),
    }

    analyzed = []
    for result in eval_results:
        metrics = {
            "faithfulness": float(result.faithfulness),
            "answer_relevancy": float(result.answer_relevancy),
            "context_precision": float(result.context_precision),
            "context_recall": float(result.context_recall),
        }
        metrics = {
            name: value if math.isfinite(value) else 0.0
            for name, value in metrics.items()
        }
        average_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "contexts": list(result.contexts),
            "average_score": average_score,
            "worst_metric": worst_metric,
            "score": metrics[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda item: (item["average_score"], item["score"]))
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
