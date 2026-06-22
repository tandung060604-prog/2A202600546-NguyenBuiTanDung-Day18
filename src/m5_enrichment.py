from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _extract_json_object(content: str) -> dict:
    """Parse a JSON object, including responses wrapped in markdown fences."""
    if not content:
        return {}
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _fallback_summary(text: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]
    return " ".join(sentences[:2]) if sentences else text.strip()


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    if n_questions <= 0:
        return []
    sentences = [
        sentence.strip().rstrip(".!?")
        for sentence in re.split(r"[.!?\n]+", text)
        if len(sentence.strip()) > 10
    ]
    return [f"{sentence}?" for sentence in sentences[:n_questions]]


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    category_keywords = {
        "hr": ("nhân viên", "nghỉ phép", "lương", "thử việc", "bảo hiểm"),
        "it": ("mật khẩu", "vpn", "mfa", "cntt", "malware"),
        "finance": ("chi phí", "tạm ứng", "thanh toán", "vnđ", "ngân sách"),
    }
    category = "policy"
    for candidate, keywords in category_keywords.items():
        if any(keyword in lowered for keyword in keywords):
            category = candidate
            break

    topic_source = _fallback_summary(text)
    topic = topic_source[:120].strip() or "general"
    return {
        "topic": topic,
        "entities": [],
        "category": category,
        "language": "vi",
    }


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tóm tắt đoạn văn trong tối đa 2 câu ngắn gọn bằng "
                            "tiếng Việt. Giữ nguyên số liệu và quy định quan trọng."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=150,
            )
            summary = (response.choices[0].message.content or "").strip()
            if summary:
                return summary
        except Exception as error:
            print(f"  ⚠️  OpenAI summarize failed: {error}")
    return _fallback_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if not text.strip() or n_questions <= 0:
        return []
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Tạo đúng {n_questions} câu hỏi tiếng Việt mà đoạn "
                            "văn có thể trả lời. Mỗi câu trên một dòng, không đánh số."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=200,
            )
            content = (response.choices[0].message.content or "").strip()
            questions = []
            for line in content.splitlines():
                question = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                if question:
                    questions.append(
                        question if question.endswith("?") else f"{question}?"
                    )
            if questions:
                return questions[:n_questions]
        except Exception as error:
            print(f"  ⚠️  OpenAI HyQA failed: {error}")
    return _fallback_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text:
        return text
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết đúng một câu tiếng Việt mô tả ngữ cảnh và chủ "
                            "đề của đoạn trích. Không thêm thông tin không có."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                temperature=0,
                max_tokens=80,
            )
            context = (response.choices[0].message.content or "").strip()
            if context:
                return f"{context}\n\n{text}"
        except Exception as error:
            print(f"  ⚠️  OpenAI contextual failed: {error}")

    prefix = (
        f"Đoạn trích từ tài liệu {document_title}."
        if document_title
        else "Đoạn trích từ tài liệu chính sách."
    )
    return f"{prefix}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    fallback = _fallback_metadata(text)
    if not text.strip() or not OPENAI_API_KEY:
        return fallback
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Trích xuất metadata và chỉ trả về JSON với schema: "
                        '{"topic":"...", "entities":["..."], '
                        '"category":"policy|hr|it|finance", "language":"vi|en"}.'
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=180,
        )
        metadata = _extract_json_object(
            response.choices[0].message.content or ""
        )
        return {**fallback, **metadata} if metadata else fallback
    except Exception as error:
        print(f"  ⚠️  OpenAI metadata failed: {error}")
        return fallback


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    fallback = {
        "summary": _fallback_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": (
            f"Đoạn trích từ tài liệu {source}."
            if source
            else "Đoạn trích từ tài liệu chính sách."
        ),
        "metadata": _fallback_metadata(text),
    }
    if not text.strip() or not OPENAI_API_KEY:
        return fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Phân tích đoạn văn và chỉ trả về một JSON object gồm: "
                        '"summary" (tối đa 2 câu), "questions" (đúng 3 câu hỏi), '
                        '"context" (1 câu ngữ cảnh), và "metadata" với topic, '
                        "entities, category (policy|hr|it|finance), language."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=400,
        )
        result = _extract_json_object(response.choices[0].message.content or "")
        if not result:
            return fallback

        questions = result.get("questions", fallback["questions"])
        if not isinstance(questions, list):
            questions = fallback["questions"]
        metadata = result.get("metadata", fallback["metadata"])
        if not isinstance(metadata, dict):
            metadata = fallback["metadata"]
        return {
            "summary": str(result.get("summary") or fallback["summary"]),
            "questions": [str(question) for question in questions[:3]],
            "context": str(result.get("context") or fallback["context"]),
            "metadata": {**fallback["metadata"], **metadata},
        }
    except Exception as error:
        print(f"  ⚠️  Enrichment API failed: {error}")
        return fallback


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
