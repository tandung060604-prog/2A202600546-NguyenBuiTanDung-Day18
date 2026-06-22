from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _split_sentences(text: str) -> list[str]:
    """Split prose into non-empty units while preserving markdown headers."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
        if sentence.strip()
    ]


def _split_to_size(text: str, max_size: int) -> list[str]:
    """Split text without dropping content, keeping every piece within max_size."""
    if not text.strip():
        return []
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")

    pieces: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        remaining = sentence
        while len(remaining) > max_size:
            if current:
                pieces.append(current.strip())
                current = ""
            split_at = remaining.rfind(" ", 0, max_size + 1)
            if split_at <= 0:
                split_at = max_size
            pieces.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()

        if not remaining:
            continue
        candidate = f"{current} {remaining}".strip()
        if current and len(candidate) > max_size:
            pieces.append(current.strip())
            current = remaining
        else:
            current = candidate

    if current:
        pieces.append(current.strip())
    return pieces


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.asarray(model.encode(sentences), dtype=float)
    groups: list[list[str]] = [[sentences[0]]]

    for index in range(1, len(sentences)):
        previous = embeddings[index - 1]
        current = embeddings[index]
        similarity = float(
            np.dot(previous, current)
            / (np.linalg.norm(previous) * np.linalg.norm(current) + 1e-9)
        )
        if similarity < threshold:
            groups.append([sentences[index]])
        else:
            groups[-1].append(sentences[index])

    return [
        Chunk(
            text="\n\n".join(group),
            metadata={
                **metadata,
                "strategy": "semantic",
                "chunk_index": index,
            },
        )
        for index, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be greater than 0")

    paragraphs = [
        piece
        for paragraph in text.split("\n\n")
        for piece in _split_to_size(paragraph, parent_size)
    ]
    if not paragraphs:
        return ([], [])

    parent_texts: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        separator_length = 2 if current else 0
        if current and current_length + separator_length + len(paragraph) > parent_size:
            parent_texts.append("\n\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
        else:
            current.append(paragraph)
            current_length += separator_length + len(paragraph)
    if current:
        parent_texts.append("\n\n".join(current))

    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = str(metadata.get("source", "document"))
    source_id = re.sub(r"[^A-Za-z0-9_-]+", "_", source).strip("_") or "document"

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{source_id}_parent_{parent_index}"
        parents.append(
            Chunk(
                text=parent_text,
                metadata={
                    **metadata,
                    "strategy": "hierarchical",
                    "chunk_type": "parent",
                    "chunk_index": parent_index,
                    "parent_id": parent_id,
                },
            )
        )
        for child_index, child_text in enumerate(_split_to_size(parent_text, child_size)):
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "strategy": "hierarchical",
                        "chunk_type": "child",
                        "chunk_index": child_index,
                        "parent_id": parent_id,
                    },
                    parent_id=parent_id,
                )
            )

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    header_pattern = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
    chunks: list[Chunk] = []
    current_header = ""
    current_section = "Preamble"
    current_lines: list[str] = []

    def flush_section() -> None:
        body = "\n".join(current_lines).strip()
        chunk_text = "\n\n".join(part for part in (current_header, body) if part)
        if not chunk_text:
            return
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": current_section,
                    "chunk_index": len(chunks),
                },
            )
        )

    for line in text.splitlines():
        match = header_pattern.match(line)
        if match:
            flush_section()
            current_header = line.strip()
            current_section = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush_section()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
