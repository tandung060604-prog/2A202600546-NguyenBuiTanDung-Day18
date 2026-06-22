from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        # Keep retrieval usable if the optional Vietnamese tokenizer fails.
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(chunk.get("text", "")).lower().split()
            for chunk in self.documents
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        results = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            document = self.documents[index]
            results.append(
                SearchResult(
                    text=document.get("text", ""),
                    score=score,
                    metadata=dict(document.get("metadata", {})),
                    method="bm25",
                )
            )
            if len(results) >= top_k:
                break
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        documents = list(chunks)
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not documents:
            return

        texts = [document.get("text", "") for document in documents]
        vectors = self._get_encoder().encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 32,
            normalize_embeddings=True,
        )
        points = [
            PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={
                    **document.get("metadata", {}),
                    "text": document.get("text", ""),
                },
            )
            for index, (document, vector) in enumerate(zip(documents, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query.strip() or top_k <= 0:
            return []

        query_vector = self._get_encoder().encode(
            query,
            normalize_embeddings=True,
        ).tolist()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            results.append(
                SearchResult(
                    text=text,
                    score=float(point.score),
                    metadata=payload,
                    method="dense",
                )
            )
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []

    fused: dict[str, dict] = {}
    for results in results_list:
        seen_in_list = set()
        for rank, result in enumerate(results):
            # A document should contribute only once per retriever.
            if result.text in seen_in_list:
                continue
            seen_in_list.add(result.text)
            entry = fused.setdefault(
                result.text,
                {"score": 0.0, "result": result, "best_rank": rank},
            )
            entry["score"] += 1.0 / (k + rank + 1)
            entry["best_rank"] = min(entry["best_rank"], rank)

    ranked = sorted(
        fused.values(),
        key=lambda entry: (
            -entry["score"],
            entry["best_rank"],
            entry["result"].text,
        ),
    )
    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=dict(entry["result"].metadata),
            method="hybrid",
        )
        for entry in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
