import re

import numpy as np
import pytest

from query_adaptive_rag.expansion.context_expander import (
    ExpansionRule,
    TokenBudgetError,
    assemble_context,
    expand_context,
)
from query_adaptive_rag.retrieval.dense import DenseRetriever, SearchHit, SearchResponse


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, verbose=True):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return result


class LookupEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, **kwargs):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def make_chunk(chunk_id, text, start, end, prev=None, next_=None, doc_id="doc-a"):
    return {
        "chunk_id": chunk_id, "doc_id": doc_id, "text": text, "start": start, "end": end,
        "prev_chunk_id": prev, "next_chunk_id": next_, "token_count": len(text.split()), "method": "semantic",
    }


def test_faiss_retrieval_top_k_scores_and_stable_ids():
    chunks = [
        make_chunk("chunk-z", "alpha text", 0, 10),
        make_chunk("chunk-a", "beta text", 11, 20),
        make_chunk("chunk-b", "gamma text", 21, 31),
    ]
    encoder = LookupEncoder({
        "query": [1.0, 0.0],
        "alpha text": [1.0, 0.0],
        "beta text": [1.0, 0.0],
        "gamma text": [0.0, 1.0],
    })
    retriever = DenseRetriever(encoder)
    retriever.build(chunks)
    first = retriever.retrieve("query", top_k=2)
    second = retriever.retrieve("query", top_k=2)

    assert [hit.chunk["chunk_id"] for hit in first.hits] == ["chunk-a", "chunk-z"]
    assert [hit.chunk["chunk_id"] for hit in second.hits] == ["chunk-a", "chunk-z"]
    assert [hit.chunk["doc_id"] for hit in first.hits] == ["doc-a", "doc-a"]
    assert all(hit.score == pytest.approx(1.0) for hit in first.hits)
    assert first.latency_ms >= 0


def test_retriever_rejects_duplicate_chunk_ids():
    chunk = make_chunk("same", "alpha", 0, 5)
    retriever = DenseRetriever(LookupEncoder({"alpha": [1.0, 0.0]}))
    with pytest.raises(ValueError, match="unique"):
        retriever.build([chunk, dict(chunk)])


def test_expansion_uses_only_relevant_same_document_neighbors_and_source_order():
    chunks = [
        make_chunk("irrelevant", "Irrelevant text.", 0, 16, next_="left"),
        make_chunk("left", "Left evidence.", 17, 31, prev="irrelevant", next_="seed"),
        make_chunk("seed", "Seed evidence.", 32, 46, prev="left", next_="right"),
        make_chunk("right", "Right evidence.", 47, 62, prev="seed", next_="far"),
        make_chunk("far", "Far evidence.", 63, 76, prev="right"),
        make_chunk("other-doc", "Foreign evidence.", 0, 17, prev="seed", doc_id="doc-b"),
    ]
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    seed = by_id["seed"]
    search = SearchResponse([SearchHit(seed, 0.95, 1)], np.array([1.0, 0.0], dtype=np.float32), 1.5)
    scores = {"irrelevant": 0.10, "left": 0.75, "right": 0.90, "far": 0.80, "other-doc": 0.99}
    result = expand_context(
        search, "multi_hop", chunks, TinyTokenizer(), lambda _q, chunk_id: scores[chunk_id],
        {"multi_hop": ExpansionRule(max_depth=2, max_expansions=3)}, 30, 0.2,
    )

    assert result.seed_chunk_ids == ["seed"]
    assert set(result.expanded_chunk_ids) == {"left", "right", "far"}
    assert len(result.expanded_chunk_ids) == len(set(result.expanded_chunk_ids))
    assert "other-doc" not in result.expanded_chunk_ids
    assert "irrelevant" not in result.expanded_chunk_ids
    assert result.similarity_scores["irrelevant"] < 0.2
    assert result.context_text.index("Left") < result.context_text.index("Seed") < result.context_text.index("Right") < result.context_text.index("Far")
    assert result.context_token_count <= result.token_budget


def test_exact_duplicate_neighbor_text_is_not_added_twice():
    seed = make_chunk("seed", "Same evidence.", 0, 14, next_="duplicate")
    duplicate = make_chunk("duplicate", "Same evidence.", 20, 34, prev="seed")
    chunks = [seed, duplicate]
    search = SearchResponse([SearchHit(seed, 1.0, 1)], np.array([1.0, 0.0]), 0.1)
    result = expand_context(
        search, "multi_concept", chunks, TinyTokenizer(), lambda *_: 0.9,
        {"multi_concept": ExpansionRule(1, 1)}, 20, 0.2,
    )

    assert result.expanded_chunk_ids == []
    assert result.deduplicated_chunk_ids == ["duplicate"]
    assert result.context_text.count("Same evidence.") == 1


def test_overlap_is_rendered_once_and_source_order_is_stable():
    later = make_chunk("later", "words tail.", 7, 18, prev="first")
    first = make_chunk("first", "shared words", 0, 12, next_="later")
    rendered, _ = assemble_context([later, first], TinyTokenizer())
    assert rendered == "shared words\n\n tail."
    assert rendered.count("shared") == 1


def test_budget_exhaustion_rejects_expansion_and_never_overflows():
    seed = make_chunk("seed", "Seed words.", 0, 11, next_="neighbor")
    neighbor = make_chunk("neighbor", "Neighbor words.", 12, 27, prev="seed")
    chunks = [seed, neighbor]
    search = SearchResponse([SearchHit(seed, 1.0, 1)], np.array([1.0, 0.0]), 0.1)
    seed_tokens = len(seed["text"].split())
    result = expand_context(
        search, "multi_hop", chunks, TinyTokenizer(), lambda *_: 0.9,
        {"multi_hop": ExpansionRule(1, 2)}, seed_tokens, 0.2,
    )

    assert result.context_token_count == seed_tokens
    assert result.context_token_count <= result.token_budget
    assert result.expanded_chunk_ids == []
    assert result.budget_rejected_chunk_ids == ["neighbor"]


def test_seed_context_over_budget_fails_explicitly():
    seed = make_chunk("seed", "Seed has multiple words.", 0, 23)
    search = SearchResponse([SearchHit(seed, 1.0, 1)], np.array([1.0, 0.0]), 0.1)
    with pytest.raises(TokenBudgetError, match="exceeding budget"):
        expand_context(
            search, "single_fact", [seed], TinyTokenizer(), lambda *_: 0.5,
            {"single_fact": ExpansionRule(0, 0)}, 1, 0.2,
        )


def test_baseline_and_proposed_method_receive_identical_seed_ids():
    seed = make_chunk("seed", "Seed context.", 0, 13, next_="neighbor")
    neighbor = make_chunk("neighbor", "More context.", 14, 28, prev="seed")
    search = SearchResponse([SearchHit(seed, 0.9, 1)], np.array([1.0, 0.0]), 0.2)
    baseline_text, _ = assemble_context([hit.chunk for hit in search.hits], TinyTokenizer())
    result = expand_context(
        search, "single_fact", [seed, neighbor], TinyTokenizer(), lambda *_: 0.9,
        {"single_fact": ExpansionRule(0, 0)}, 20, 0.2,
    )
    assert result.seed_chunk_ids == [hit.chunk["chunk_id"] for hit in search.hits]
    assert result.expanded_chunk_ids == []
    assert result.context_text == baseline_text
