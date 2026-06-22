from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
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
    zero_result = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }
    if not questions or not answers or not contexts or not ground_truths:
        return zero_result
    if os.getenv("ENABLE_RAGAS") != "1":
        return zero_result
    if not os.getenv("OPENAI_API_KEY"):
        return zero_result
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        from config import LLM_API_KEY, LLM_BASE_URL, ANSWER_MODEL
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(
            model=ANSWER_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL or None,
        )
        embeddings = OpenAIEmbeddings(
            model="text-embedding-004" if "googleapis.com" in (LLM_BASE_URL or "") else "text-embedding-3-small",
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL or None,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": float(df.get("faithfulness", []).mean() if "faithfulness" in df else 0.0),
            "answer_relevancy": float(df.get("answer_relevancy", []).mean() if "answer_relevancy" in df else 0.0),
            "context_precision": float(df.get("context_precision", []).mean() if "context_precision" in df else 0.0),
            "context_recall": float(df.get("context_recall", []).mean() if "context_recall" in df else 0.0),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zero_result


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt and reduce unsupported generation"),
        "context_recall": ("Missing relevant chunks", "Improve chunking, hybrid retrieval, or recall depth"),
        "context_precision": ("Too many irrelevant chunks", "Add stronger reranking or metadata filtering"),
        "answer_relevancy": ("Answer misses the user intent", "Improve answer prompt and query understanding"),
    }

    analyzed = []
    for result in eval_results:
        metrics = {
            "faithfulness": float(result.faithfulness),
            "answer_relevancy": float(result.answer_relevancy),
            "context_precision": float(result.context_precision),
            "context_recall": float(result.context_recall),
        }
        avg_score = sum(metrics.values()) / 4
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return sorted(analyzed, key=lambda item: item["score"])[:bottom_n]


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
