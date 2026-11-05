"""Token-window chunking with overlap and source metadata."""

from __future__ import annotations

from typing import Any

from .text_utils import OffsetTokenizer


def fixed_size_chunks(
    text: str,
    doc_id: str,
    tokenizer: OffsetTokenizer,
    chunk_size: int,
    overlap_tokens: int = 32,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size:
        raise ValueError("overlap_tokens must be >= 0 and smaller than chunk_size")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    chunks = []
    step = chunk_size - overlap_tokens
    for start_token in range(0, len(ids), step):
        end_token = min(start_token + chunk_size, len(ids))
        if start_token >= end_token:
            break
        start_char = int(offsets[start_token][0])
        end_char = int(offsets[end_token - 1][1])
        chunk_text = text[start_char:end_char]
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}::fixed{chunk_size}::{len(chunks):05d}",
            "text": chunk_text,
            "token_count": end_token - start_token,
            "start": start_char,
            "end": end_char,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "method": f"fixed_{chunk_size}",
        })
        if end_token == len(ids):
            break
    _link_neighbors(chunks)
    return chunks


def _link_neighbors(chunks: list[dict[str, Any]]) -> None:
    for index, chunk in enumerate(chunks):
        if index:
            chunk["prev_chunk_id"] = chunks[index - 1]["chunk_id"]
        if index + 1 < len(chunks):
            chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"]
