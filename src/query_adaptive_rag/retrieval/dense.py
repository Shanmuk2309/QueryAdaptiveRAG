"""Deterministic cosine dense retrieval using a FAISS CPU flat index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np


@dataclass(frozen=True)
class SearchHit:
    chunk: dict[str, Any]
    score: float
    rank: int


@dataclass(frozen=True)
class SearchResponse:
    hits: list[SearchHit]
    query_embedding: np.ndarray
    latency_ms: float


class DenseRetriever:
    """Encode chunks once, then retrieve with normalized inner-product scores."""

    def __init__(self, encoder: Any, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = encoder
        self.model_name = model_name
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict[str, Any]] = []
        self.vectors: np.ndarray | None = None
        self._positions: dict[str, int] = {}

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("embeddings must be a finite 2D matrix")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero-length embeddings cannot be indexed")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def _encode(self, texts: list[str]) -> np.ndarray:
        raw = self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False,
        )
        return self._normalize(np.asarray(raw))

    def build(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("at least one chunk is required to build an index")
        ids = [chunk.get("chunk_id") for chunk in chunks]
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in ids):
            raise ValueError("each chunk must have a non-empty chunk_id")
        if len(set(ids)) != len(ids):
            raise ValueError("chunk_id values must be unique")
        if any(not chunk.get("doc_id") or not chunk.get("text") for chunk in chunks):
            raise ValueError("each chunk must include doc_id and text")

        vectors = self._encode([chunk["text"] for chunk in chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = [dict(chunk) for chunk in chunks]
        self.vectors = vectors
        self._positions = {chunk["chunk_id"]: i for i, chunk in enumerate(self.chunks)}

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        return self._encode([query])

    def retrieve(self, query: str, top_k: int = 5) -> SearchResponse:
        if self.index is None or self.vectors is None:
            raise RuntimeError("build the FAISS index before retrieval")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        started = perf_counter()
        query_vector = self.embed_query(query)
        # Search all rows so ties at the top-k boundary can be ordered deterministically.
        scores, positions = self.index.search(query_vector, len(self.chunks))
        ranked = [
            (float(score), int(position))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]
        ranked.sort(key=lambda pair: (-pair[0], self.chunks[pair[1]]["chunk_id"]))
        hits = [
            SearchHit(dict(self.chunks[position]), score, rank)
            for rank, (score, position) in enumerate(ranked[:min(top_k, len(ranked))], start=1)
        ]
        latency_ms = (perf_counter() - started) * 1000.0
        return SearchResponse(hits, query_vector[0].copy(), latency_ms)

    def score(self, query_embedding: np.ndarray, chunk_id: str) -> float:
        """Score an indexed chunk against an already normalized query vector."""
        if self.vectors is None:
            raise RuntimeError("build the FAISS index before scoring")
        position = self._positions.get(chunk_id)
        if position is None:
            raise KeyError(f"unknown chunk_id: {chunk_id}")
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        vector = self._normalize(vector)
        if vector.shape[1] != self.vectors.shape[1]:
            raise ValueError("query embedding dimension differs from indexed vectors")
        return float(np.dot(vector[0], self.vectors[position]))

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("build the FAISS index before saving")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output))
