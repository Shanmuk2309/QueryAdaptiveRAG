import numpy as np
import pytest

from query_adaptive_rag.chunking.fixed_size import fixed_size_chunks
from query_adaptive_rag.chunking.semantic import semantic_chunks
from query_adaptive_rag.chunking.text_utils import count_tokens, segment_sentences


class TinyTokenizer:
    """Whitespace tokenizer with Hugging Face-style offsets for deterministic tests."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(__import__("re").finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return result


class FakeModel:
    model_name_or_path = "unit-test-model"

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            vectors.append([1.0, 0.0] if "alpha" in sentence.lower() else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_fixed_chunk_sizes_overlap_offsets_and_links():
    text = "one two three four five six seven eight nine ten"
    chunks = fixed_size_chunks(text, "doc-1", TinyTokenizer(), chunk_size=4, overlap_tokens=1)
    assert [chunk["token_count"] for chunk in chunks] == [4, 4, 4]
    assert chunks[0]["text"] == "one two three four"
    assert chunks[1]["text"].startswith("four")
    assert text[chunks[0]["start"]:chunks[0]["end"]] == chunks[0]["text"]
    assert chunks[0]["next_chunk_id"] == chunks[1]["chunk_id"]
    assert chunks[1]["prev_chunk_id"] == chunks[0]["chunk_id"]
    assert chunks[-1]["next_chunk_id"] is None


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (4, 4), (4, -1)])
def test_fixed_chunker_rejects_invalid_sizes(chunk_size, overlap):
    with pytest.raises(ValueError):
        fixed_size_chunks("text", "doc", TinyTokenizer(), chunk_size, overlap)


def test_semantic_chunking_preserves_metadata_and_splits_low_similarity():
    text = "alpha begins here. alpha continues here. beta topic starts now. beta continues now."
    chunks = semantic_chunks(
        text, "doc-2", TinyTokenizer(), FakeModel(), similarity_threshold=0.8,
        min_tokens=1, max_tokens=20,
    )
    assert len(chunks) == 2
    assert chunks[0]["text"] == "alpha begins here. alpha continues here."
    assert chunks[1]["text"] == "beta topic starts now. beta continues now."
    assert chunks[0]["token_count"] == count_tokens(chunks[0]["text"], TinyTokenizer())
    assert chunks[0]["prev_chunk_id"] is None
    assert chunks[0]["next_chunk_id"] == chunks[1]["chunk_id"]
    for chunk in chunks:
        assert text[chunk["start"]:chunk["end"]] == chunk["text"]


def test_semantic_chunker_enforces_hard_maximum():
    text = "alpha one. alpha two. alpha three. alpha four."
    chunks = semantic_chunks(text, "doc", TinyTokenizer(), FakeModel(), 0.0, 1, 5)
    assert all(chunk["token_count"] <= 5 for chunk in chunks)


def test_semantic_short_source_document_is_retained_below_minimum():
    text = "alpha only."
    chunks = semantic_chunks(text, "short-doc", TinyTokenizer(), FakeModel(), 0.8, 80, 400)

    assert len(chunks) == 1
    assert chunks[0]["token_count"] < 80
    assert chunks[0]["text"] == text
    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == len(text)
    assert chunks[0]["prev_chunk_id"] is None
    assert chunks[0]["next_chunk_id"] is None


def test_semantic_short_trailing_chunk_merges_without_losing_id_or_text():
    text = "alpha one two three. beta x."
    chunks = semantic_chunks(text, "tail-doc", TinyTokenizer(), FakeModel(), 0.8, 4, 10)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "tail-doc::semantic::00000"
    assert chunks[0]["merged_chunk_ids"] == ["tail-doc::semantic::00001"]
    assert chunks[0]["text"] == text
    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == len(text)
    assert chunks[0]["token_count"] == count_tokens(text, TinyTokenizer())
    assert chunks[0]["prev_chunk_id"] is None
    assert chunks[0]["next_chunk_id"] is None


def test_semantic_short_tail_stays_when_merge_would_exceed_maximum():
    text = "alpha one two three. beta x."
    chunks = semantic_chunks(text, "bounded-doc", TinyTokenizer(), FakeModel(), 0.8, 4, 5)

    assert len(chunks) == 2
    assert chunks[0]["token_count"] + count_tokens(" beta x.", TinyTokenizer()) > 5
    assert chunks[0]["next_chunk_id"] == chunks[1]["chunk_id"]
    assert chunks[1]["prev_chunk_id"] == chunks[0]["chunk_id"]
    assert text[chunks[0]["start"]:chunks[0]["end"]] == chunks[0]["text"]
    assert text[chunks[1]["start"]:chunks[1]["end"]] == chunks[1]["text"]


def test_sentence_segmentation_retains_offsets():
    text = "  First sentence.\nSecond sentence!  "
    parts = segment_sentences(text)
    assert [part.text for part in parts] == ["First sentence.", "Second sentence!"]
    assert [text[p.start:p.end] for p in parts] == [p.text for p in parts]
