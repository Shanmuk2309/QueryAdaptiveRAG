"""Deterministic integration checks for the backend's M4-v3 adapter.

These tests exercise the existing research classifier/parser/expander and do
not alter its thresholds or behavior.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

from query_adaptive_rag.classification.complexity import classify_query
from query_adaptive_rag.expansion.context_expander import ExpansionRule
from query_adaptive_rag.expansion.context_expander_v3 import expand_context_v3
from query_adaptive_rag.expansion.query_facets import extract_facts, parse_query_frame
from query_adaptive_rag.retrieval.dense import SearchHit, SearchResponse
from app.services import retrieval_service as retrieval_module
from app.services.retrieval_service import RetrievalService


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_kwargs):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result


def chunk(chunk_id: str, text: str, start: int, *, prev: str | None = None,
          next_: str | None = None) -> dict:
    return {"chunk_id": chunk_id, "doc_id": "doc-1", "text": text,
            "start": start, "end": start + len(text), "prev_chunk_id": prev,
            "next_chunk_id": next_, "token_count": len(text.split()), "method": "semantic"}


@pytest.mark.parametrize("query,expected,relation", [
    ("Who wrote Arizona?", "single_fact", "authored_by"),
    ("Between Gold Center and Sarcobatus Flat, which is farther north?", "multi_concept", "comparison_property"),
    ("Which film was directed by Harold Ramis and stars Bill Murray?", "multi_hop", "cast_member"),
])
def test_query_classification_and_frame_for_supported_query_shapes(query, expected, relation):
    assert classify_query(query)["complexity"] == expected
    frame = parse_query_frame(query)
    assert frame.entities
    assert relation in {request.relation for request in frame.relations}


def test_conceptual_causal_query_exposes_current_parser_limitation():
    # This is a known limit of the existing conservative parser: it extracts
    # named concepts but has no generic causal/relationship predicate mapping.
    frame = parse_query_frame("What connects Climate Change and Global Warming?")
    assert {"climate change", "global warming"}.issubset(set(frame.entities))
    assert frame.relations == ()


def run_expander(query: str, seed_text: str, neighbor_text: str):
    seed = chunk("seed", seed_text, 0, next_="neighbor")
    neighbor = chunk("neighbor", neighbor_text, len(seed_text) + 1, prev="seed")
    search = SearchResponse([SearchHit(seed, 0.9, 1)], np.asarray([1.0, 0.0], dtype=np.float32), 0.1)
    vectors = {"seed": np.asarray([1.0, 0.0], dtype=np.float32),
               "neighbor": np.asarray([0.8, 0.6], dtype=np.float32)}
    result = expand_context_v3(
        search, query, classify_query(query)["complexity"], [seed, neighbor], vectors,
        TinyTokenizer(), lambda _query_vector, _chunk_id: 0.9,
        {"single_fact": ExpansionRule(0, 0), "multi_concept": ExpansionRule(1, 2),
         "multi_hop": ExpansionRule(2, 4)}, 100,
    )
    return result, [seed, neighbor]


def test_no_neighbor_with_supported_query_fact_is_rejected():
    result, _ = run_expander(
        "Which film was directed by Alex Smith and stars Pat Jones?",
        "Example Film was directed by Alex Smith.",
        "The nearby theatre hosted a summer festival.",
    )
    assert result.expanded_chunk_ids == []
    assert result.candidate_decisions[0]["candidate_facts"] == []
    assert result.candidate_decisions[0]["reason"] == "no_supported_query_fact"


def test_relevant_neighbor_fact_is_extracted_matched_and_expanded():
    query = "Which film was directed by Harold Ramis and stars Bill Murray?"
    result, chunks = run_expander(
        query, "Groundhog Day was directed by Harold Ramis.",
        "Groundhog Day also stars Bill Murray.",
    )
    extracted = extract_facts(chunks)
    assert ("groundhog day", "cast_member", "bill murray") in {fact.key() for fact in extracted}
    assert result.expanded_chunk_ids == ["neighbor"]
    assert result.candidate_decisions[0]["reason"] == "adds_new_query_fact"
    assert result.candidate_decisions[0]["supporting_fact_chain"]


def test_backend_passes_original_query_and_method_chunks_to_research_expander(monkeypatch):
    query = "Which film was directed by Harold Ramis and stars Bill Murray?"
    seed = chunk("seed", "Groundhog Day was directed by Harold Ramis.", 0,
                 next_="neighbor")
    neighbor = chunk("neighbor", "Groundhog Day also stars Bill Murray.", 50, prev="seed")
    search = SearchResponse([SearchHit(seed, 0.9, 1)], np.asarray([1.0, 0.0], dtype=np.float32), 1.0)

    class Retriever:
        def retrieve(self, received_query, _top_k):
            assert received_query == query
            return search

        def score(self, *_args):
            return 0.9

    runtime = SimpleNamespace(
        indexes={"M4-v3": Retriever()},
        chunks_by_method={"M4-v3": [seed, neighbor]},
        vectors_by_method={"M4-v3": {"seed": np.array([1., 0.]), "neighbor": np.array([.8, .6])}},
        tokenizer=TinyTokenizer(),
    )
    captured = {}

    def fake_expander(search_arg, query_arg, complexity, chunks, vectors, tokenizer,
                      score_chunk, rules, budget):
        captured.update(query=query_arg, complexity=complexity, chunks=chunks, vectors=vectors)
        return SimpleNamespace(expanded_chunk_ids=[], similarity_scores={}, context_text=seed["text"],
                               context_token_count=7, latency_ms=0.0, candidate_decisions=[])

    monkeypatch.setattr(retrieval_module, "expand_context_v3", fake_expander)
    service = RetrievalService(SimpleNamespace(context_token_budget=100), runtime)
    service.query(query, "M4-v3", 5)

    assert captured["query"] == query
    assert captured["complexity"] == "multi_hop"
    assert [item["text"] for item in captured["chunks"]] == [seed["text"], neighbor["text"]]

