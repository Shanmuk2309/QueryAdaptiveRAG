"""Tokenizer-aware validation of chunk inputs against an encoder limit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def encoder_content_token_limit(tokenizer: Any, max_sequence_length: int) -> tuple[int, int]:
    """Return (content-token limit, special-token count) for one input sequence."""
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_limit = max_sequence_length - special_tokens
    if content_limit <= 0:
        raise ValueError(
            f"encoder sequence limit {max_sequence_length} leaves no content tokens "
            f"after {special_tokens} special tokens"
        )
    return content_limit, special_tokens


def audit_chunk_embedding_lengths(
    chunks: Sequence[Mapping[str, Any]], tokenizer: Any, max_sequence_length: int,
) -> dict[str, int]:
    """Check actual encoded input lengths; raise before any over-limit text is embedded."""
    content_limit, special_tokens = encoder_content_token_limit(tokenizer, max_sequence_length)
    over_limit: list[tuple[str, int]] = []
    maximum = 0
    for chunk in chunks:
        encoded = tokenizer(
            str(chunk["text"]), add_special_tokens=True, truncation=False,
            return_attention_mask=False, return_token_type_ids=False,
        )
        sequence_length = len(encoded["input_ids"])
        maximum = max(maximum, sequence_length)
        if sequence_length > max_sequence_length:
            over_limit.append((str(chunk.get("chunk_id", "<unknown>")), sequence_length))
    if over_limit:
        preview = ", ".join(f"{chunk_id}={length}" for chunk_id, length in over_limit[:5])
        raise ValueError(
            f"{len(over_limit)} chunk(s) exceed encoder sequence limit {max_sequence_length}: {preview}"
        )
    return {
        "encoder_max_sequence_length": max_sequence_length,
        "tokenizer_special_token_count": special_tokens,
        "max_content_tokens": content_limit,
        "checked_chunk_count": len(chunks),
        "max_observed_sequence_tokens": maximum,
        "over_limit_chunk_count": 0,
        "truncation_risk_chunk_count": 0,
    }
