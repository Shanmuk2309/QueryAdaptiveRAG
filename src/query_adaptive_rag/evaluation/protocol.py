"""Shared four-method settings used by the Day 3 comparison."""

from __future__ import annotations


METHODS = ("M1_fixed_256", "M2_fixed_512", "M3_semantic", "M4_adaptive")


def method_protocol(
    model_name: str, top_k: int, token_budget: int, fixed_overlap_tokens: int,
    embedding_max_sequence_length: int = 512,
    m2_content_tokens: int = 512,
    tokenizer_special_token_count: int | None = None,
) -> dict[str, dict]:
    """Describe the controlled retrieval inputs and method-specific chunking."""
    shared = {
        "embedding_model": model_name,
        "embedding_max_sequence_length": embedding_max_sequence_length,
        "top_k": top_k,
        "token_budget": token_budget,
    }
    return {
        "M1_fixed_256": {**shared, "chunking": "fixed", "chunk_size": 256, "overlap_tokens": fixed_overlap_tokens},
        "M2_fixed_512": {
            **shared, "chunking": "fixed", "chunk_size": m2_content_tokens,
            "overlap_tokens": fixed_overlap_tokens,
            **({"tokenizer_special_token_count": tokenizer_special_token_count,
                "chunk_size_unit": "content_tokens"} if tokenizer_special_token_count is not None else {}),
        },
        "M3_semantic": {**shared, "chunking": "semantic"},
        "M4_adaptive": {**shared, "chunking": "semantic", "seed_method": "M3_semantic"},
    }
