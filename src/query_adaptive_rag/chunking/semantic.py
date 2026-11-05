"""Sentence-aware semantic segmentation using cached sentence embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .fixed_size import _link_neighbors
from .text_utils import OffsetTokenizer, segment_sentences


def _cached_embeddings(
    sentences: list[str], model: Any, cache_dir: str | Path | None, model_id: str | None = None
) -> np.ndarray:
    if not sentences:
        return np.empty((0, 0), dtype=np.float32)
    key_data = str(model_id or getattr(model, "model_name_or_path", "unknown-model"))
    key_data += "\0" + str(model.get_sentence_embedding_dimension())
    key_data += "\0" + "\0".join(sentences)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"sentences-{hashlib.sha256(key_data.encode('utf-8')).hexdigest()}.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return np.load(cache_path, allow_pickle=False)
    vectors = np.asarray(model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    if cache_path is not None:
        np.save(cache_path, vectors, allow_pickle=False)
    return vectors


def semantic_chunks(
    text: str,
    doc_id: str,
    tokenizer: OffsetTokenizer,
    model: Any,
    similarity_threshold: float = 0.45,
    min_tokens: int = 80,
    max_tokens: int = 400,
    cache_dir: str | Path | None = None,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Group adjacent sentences by similarity and merge eligible short tails.

    A complete document shorter than ``min_tokens`` stays intact. A short
    trailing group is merged into its predecessor when the result fits under
    ``max_tokens``; otherwise it remains short so no source text is lost.
    """
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("require 0 < min_tokens <= max_tokens")
    sentences = segment_sentences(text)
    if not sentences:
        return []
    vectors = _cached_embeddings([sentence.text for sentence in sentences], model, cache_dir, model_id)
    groups: list[tuple[int, int]] = []
    group_start = 0
    group_tokens = len(tokenizer(sentences[0].text, add_special_tokens=False)["input_ids"])
    for index in range(1, len(sentences)):
        similarity = float(np.dot(vectors[index - 1], vectors[index]))
        sentence_tokens = len(tokenizer(sentences[index].text, add_special_tokens=False)["input_ids"])
        # Always split before exceeding max_tokens. Avoid tiny groups by requiring
        # the min size only for semantic breaks, not for hard maximum boundaries.
        semantic_break = similarity < similarity_threshold and group_tokens >= min_tokens
        if group_tokens + sentence_tokens > max_tokens or semantic_break:
            groups.append((group_start, index))
            group_start = index
            group_tokens = sentence_tokens
        else:
            group_tokens += sentence_tokens
    groups.append((group_start, len(sentences)))

    chunks = []
    for first, stop in groups:
        start_char, end_char = sentences[first].start, sentences[stop - 1].end
        chunk_text = text[start_char:end_char]
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}::semantic::{len(chunks):05d}",
            "text": chunk_text,
            "token_count": len(tokenizer(chunk_text, add_special_tokens=False)["input_ids"]),
            "start": start_char,
            "end": end_char,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "method": "semantic",
        })
    if len(chunks) > 1 and chunks[-1]["token_count"] < min_tokens:
        previous, trailing = chunks[-2], chunks[-1]
        merged_text = text[previous["start"]:trailing["end"]]
        merged_token_count = len(tokenizer(merged_text, add_special_tokens=False)["input_ids"])
        if merged_token_count <= max_tokens:
            previous["text"] = merged_text
            previous["end"] = trailing["end"]
            previous["token_count"] = merged_token_count
            previous.setdefault("merged_chunk_ids", []).append(trailing["chunk_id"])
            chunks.pop()
    _link_neighbors(chunks)
    return chunks


def count_tokens(text: str, tokenizer: OffsetTokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
