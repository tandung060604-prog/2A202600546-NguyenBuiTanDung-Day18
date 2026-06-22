from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass
from hashlib import md5

import numpy as np

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
    if os.getenv("USE_HEAVY_MODELS") == "1":
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    return text.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if float(scores[i]) > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self._memory_mode = False
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(location=":memory:")
            self._memory_mode = True
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("USE_HEAVY_MODELS") == "1":
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            else:
                class HashingEncoder:
                    def encode(self, texts, show_progress_bar: bool = False):
                        if isinstance(texts, str):
                            return self._encode_one(texts)
                        return np.array([self._encode_one(text) for text in texts])

                    def _encode_one(self, text: str):
                        vector = np.zeros(EMBEDDING_DIM, dtype=float)
                        for token in text.lower().split():
                            idx = int(md5(token.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM
                            vector[idx] += 1.0
                        norm = np.linalg.norm(vector)
                        return vector if norm == 0 else vector / norm

                self._encoder = HashingEncoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        texts = [c["text"] for c in chunks]
        if not texts:
            return
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)
        points = [
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
        return [
            SearchResult(
                text=point.payload["text"],
                score=float(point.score),
                metadata={k: v for k, v in point.payload.items() if k != "text"},
                method="dense",
            )
            for point in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked
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
