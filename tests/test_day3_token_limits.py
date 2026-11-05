from pathlib import Path

import yaml
import pytest

from query_adaptive_rag.chunking.fixed_size import fixed_size_chunks
from query_adaptive_rag.evaluation.token_limits import (
    audit_chunk_embedding_lengths,
    encoder_content_token_limit,
)


class FakeTokenizer:
    def __init__(self, special_tokens=2):
        self.special_tokens = special_tokens

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return self.special_tokens

    def __call__(self, text, add_special_tokens=False, **kwargs):
        tokens = text.split()
        offsets = []
        cursor = 0
        for token in tokens:
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        ids = list(range(len(tokens)))
        if add_special_tokens:
            ids = [0] * self.special_tokens + ids
        result = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result

    def decode(self, token_ids, **kwargs):
        return " ".join(str(token) for token in token_ids)


def test_content_window_uses_tokenizer_reported_special_token_count():
    assert encoder_content_token_limit(FakeTokenizer(special_tokens=3), 512) == (509, 3)


def test_corrected_m2_chunks_have_510_content_tokens_and_32_overlap():
    tokenizer = FakeTokenizer(special_tokens=2)
    text = " ".join(f"t{i}" for i in range(1050))
    chunks = fixed_size_chunks(text, "doc", tokenizer, 510, 32)
    assert chunks[0]["token_count"] == 510
    assert chunks[1]["token_count"] == 510
    assert chunks[1]["start"] == text.index("t478")
    assert all(chunk["token_count"] <= 510 for chunk in chunks)


def test_embedding_audit_confirms_all_chunks_fit_with_special_tokens():
    tokenizer = FakeTokenizer(special_tokens=2)
    chunks = [
        {"chunk_id": "m1", "text": " ".join(f"x{i}" for i in range(256))},
        {"chunk_id": "m2", "text": " ".join(f"x{i}" for i in range(510))},
        {"chunk_id": "m3", "text": " ".join(f"x{i}" for i in range(400))},
    ]
    audit = audit_chunk_embedding_lengths(chunks, tokenizer, 512)
    assert audit["tokenizer_special_token_count"] == 2
    assert audit["max_content_tokens"] == 510
    assert audit["checked_chunk_count"] == 3
    assert audit["max_observed_sequence_tokens"] == 512
    assert audit["over_limit_chunk_count"] == audit["truncation_risk_chunk_count"] == 0


def test_embedding_audit_rejects_a_chunk_that_would_be_truncated():
    tokenizer = FakeTokenizer(special_tokens=2)
    too_long = {"chunk_id": "bad", "text": " ".join(f"x{i}" for i in range(511))}
    with pytest.raises(ValueError, match="bad=513"):
        audit_chunk_embedding_lengths([too_long], tokenizer, 512)


def test_corrected_config_is_isolated_and_keeps_shared_model_and_split():
    original = yaml.safe_load(Path("configs/day3.yaml").read_text(encoding="utf-8"))
    corrected = yaml.safe_load(Path("configs/day3_corrected.yaml").read_text(encoding="utf-8"))
    assert corrected["encoder_safe_m2"] is True
    assert corrected["embedding_max_sequence_length"] == 512
    assert corrected["processed_output_dir"] == "data/processed/day3_corrected"
    assert corrected["results_output_dir"] == "results/day3_corrected"
    assert corrected["model_name"] == original["model_name"]
    assert corrected["evaluation_size_per_dataset"] == original["evaluation_size_per_dataset"]
