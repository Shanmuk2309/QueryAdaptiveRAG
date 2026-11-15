from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from query_adaptive_rag.retrieval.dense import DenseRetriever


@dataclass
class Runtime:
    encoder: Any
    tokenizer: Any
    model_name: str
    indexes: dict[str, DenseRetriever]
    chunks_by_method: dict[str, list[dict]]
    vectors_by_method: dict[str, dict[str, np.ndarray]]

    def load_bundle(self, method: str, bundle: tuple[Any, list[dict], np.ndarray, str]) -> None:
        index, chunks, vectors, model_name = bundle
        retriever = DenseRetriever(self.encoder, model_name)
        retriever.index = index
        retriever.chunks = chunks
        retriever.vectors = np.asarray(vectors, dtype=np.float32)
        retriever._positions = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}
        self.indexes[method] = retriever
        self.chunks_by_method[method] = chunks
        self.vectors_by_method[method] = {c["chunk_id"]: retriever.vectors[i] for i, c in enumerate(chunks)}
