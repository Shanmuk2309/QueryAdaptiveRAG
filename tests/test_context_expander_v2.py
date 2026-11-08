import copy
import re

import numpy as np
import pytest

from query_adaptive_rag.expansion.context_expander import ExpansionRule
from query_adaptive_rag.expansion.context_expander_v2 import (
    DEFAULT_REDUNDANCY_THRESHOLD,
    chunk_embedding_map_from_retriever,
    expand_context_v2,
)
from query_adaptive_rag.retrieval.dense import DenseRetriever, SearchHit, SearchResponse


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, verbose=True):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result


class LookupEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, **kwargs):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def make_chunk(chunk_id, text, start, end, prev=None, next_=None, doc_id="doc"):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "start": start,
        "end": end,
        "prev_chunk_id": prev,
        "next_chunk_id": next_,
        "token_count": len(text.split()),
        "method": "semantic",
    }


def make_case(candidate_vector, candidate_text="New fact.", token_budget=20,
              redundancy_threshold=DEFAULT_REDUNDANCY_THRESHOLD):
    seed = make_chunk("seed", "Seed fact.", 0, 10, next_="neighbor")
    neighbor = make_chunk("neighbor", candidate_text, 11, 11 + len(candidate_text), prev="seed")
    chunks = [seed, neighbor]
    query_vector = np.asarray([1.0, 0.0], dtype=np.float32)
    embeddings = {
        "seed": np.asarray([0.7, np.sqrt(0.51)], dtype=np.float32),
        "neighbor": np.asarray(candidate_vector, dtype=np.float32),
    }
    seed_score = float(np.dot(query_vector, embeddings["seed"] / np.linalg.norm(embeddings["seed"])))
    search = SearchResponse([SearchHit(seed, seed_score, 1)], query_vector, 0.2)
    score_chunk = lambda query, chunk_id: float(
        np.dot(query, embeddings[chunk_id] / np.linalg.norm(embeddings[chunk_id]))
    )
    result = expand_context_v2(
        search=search,
        complexity="multi_concept",
        chunks=chunks,
        chunk_embeddings=embeddings,
        tokenizer=TinyTokenizer(),
        score_chunk=score_chunk,
        rules={"multi_concept": ExpansionRule(max_depth=1, max_expansions=2)},
        token_budget=token_budget,
        min_similarity=0.20,
        redundancy_threshold=redundancy_threshold,
    )
    return result, chunks, search


def test_relevant_novel_neighbor_is_accepted():
    result, _, _ = make_case([0.98, 0.20])
    assert result.similarity_scores["neighbor"] >= 0.20
    assert result.redundancy_scores["neighbor"] < DEFAULT_REDUNDANCY_THRESHOLD
    assert result.expanded_chunk_ids == ["neighbor"]


def test_relevant_but_redundant_neighbor_is_rejected():
    result, _, _ = make_case([0.7, np.sqrt(0.51)])
    assert result.similarity_scores["neighbor"] >= 0.20
    assert result.redundancy_scores["neighbor"] >= DEFAULT_REDUNDANCY_THRESHOLD
    assert result.expanded_chunk_ids == []
    assert result.redundancy_rejected_chunk_ids == ["neighbor"]


def test_redundancy_threshold_can_be_configured():
    result, _, _ = make_case([0.98, 0.20], redundancy_threshold=0.80)
    assert result.redundancy_scores["neighbor"] >= 0.80
    assert result.redundancy_rejected_chunk_ids == ["neighbor"]


def test_irrelevant_neighbor_is_rejected_before_redundancy_check():
    result, _, _ = make_case([0.0, 1.0])
    assert result.similarity_scores["neighbor"] < 0.20
    assert result.expanded_chunk_ids == []
    assert result.similarity_rejected_chunk_ids == ["neighbor"]
    assert "neighbor" not in result.redundancy_scores


def test_candidate_over_token_budget_is_rejected():
    result, _, _ = make_case([0.98, 0.20], "This adds many words.", token_budget=2)
    assert result.expanded_chunk_ids == []
    assert result.budget_rejected_chunk_ids == ["neighbor"]
    assert result.context_token_count == 2
    assert result.context_token_count <= result.token_budget


def test_seed_chunks_and_search_response_are_not_mutated():
    result, chunks, search = make_case([0.98, 0.20])
    before_chunks = copy.deepcopy(chunks)
    before_seed = copy.deepcopy(search.hits[0].chunk)
    before_embedding = search.query_embedding.copy()
    assert result.seed_chunk_ids == ["seed"]
    assert chunks == before_chunks
    assert search.hits[0].chunk == before_seed
    np.testing.assert_array_equal(search.query_embedding, before_embedding)


def test_m3_retrieval_results_remain_unchanged_by_v2_expansion():
    vectors = {
        "query": [1.0, 0.0],
        "Seed fact.": [0.99, 0.1],
        "New fact.": [0.9, 0.43589],
    }
    chunks = [
        make_chunk("seed", "Seed fact.", 0, 10, next_="neighbor"),
        make_chunk("neighbor", "New fact.", 11, 20, prev="seed"),
    ]
    retriever = DenseRetriever(LookupEncoder(vectors))
    retriever.build(chunks)
    before = retriever.retrieve("query", top_k=1)
    before_ids = [hit.chunk["chunk_id"] for hit in before.hits]
    before_scores = [hit.score for hit in before.hits]
    chunk_vectors = chunk_embedding_map_from_retriever(retriever)

    result = expand_context_v2(
        search=before,
        complexity="multi_concept",
        chunks=chunks,
        chunk_embeddings=chunk_vectors,
        tokenizer=TinyTokenizer(),
        score_chunk=retriever.score,
        rules={"multi_concept": ExpansionRule(1, 1)},
        token_budget=20,
        min_similarity=0.20,
    )
    after = retriever.retrieve("query", top_k=1)

    assert result.seed_chunk_ids == before_ids
    assert [hit.chunk["chunk_id"] for hit in after.hits] == before_ids
    assert [hit.score for hit in after.hits] == before_scores


def test_v2_rejects_neighbor_from_another_document():
    seed = make_chunk("seed", "Seed fact.", 0, 10, next_="foreign")
    foreign = make_chunk("foreign", "New fact.", 11, 20, prev="seed", doc_id="other-doc")
    embeddings = {"seed": np.asarray([1.0, 0.0]), "foreign": np.asarray([0.98, 0.20])}
    search = SearchResponse([SearchHit(seed, 1.0, 1)], np.asarray([1.0, 0.0]), 0.1)
    result = expand_context_v2(
        search, "multi_concept", [seed, foreign], embeddings, TinyTokenizer(),
        lambda query, chunk_id: float(np.dot(query, embeddings[chunk_id])),
        {"multi_concept": ExpansionRule(1, 1)}, 20, 0.2,
    )
    assert result.expanded_chunk_ids == []
