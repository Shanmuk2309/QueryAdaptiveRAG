import re
from pathlib import Path

import numpy as np
import yaml

from query_adaptive_rag.evaluation.day3_dev_m4v2 import (
    METHODS,
    _development_and_heldout_ids,
    _normalize_dev_rows,
    _select_development_rows,
    evaluate_query_methods,
)
from query_adaptive_rag.expansion.context_expander import ExpansionRule
from query_adaptive_rag.retrieval.dense import DenseRetriever


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


class CountingRetriever(DenseRetriever):
    def __init__(self, encoder):
        super().__init__(encoder)
        self.retrieve_calls = 0

    def retrieve(self, query, top_k=5):
        self.retrieve_calls += 1
        return super().retrieve(query, top_k)


def _chunk(chunk_id, text, start, end, prev=None, next_=None):
    return {
        "chunk_id": chunk_id,
        "doc_id": "dev-doc",
        "text": text,
        "start": start,
        "end": end,
        "prev_chunk_id": prev,
        "next_chunk_id": next_,
        "token_count": len(text.split()),
        "method": "semantic",
    }


def test_development_split_is_reproducible_disjoint_and_exact_size():
    candidate_ids = [f"id-{index}" for index in range(250)]
    first = _development_and_heldout_ids(candidate_ids, 50, 200, 142)
    second = _development_and_heldout_ids(candidate_ids, 50, 200, 142)
    assert first == second
    assert len(first[0]) == 50
    assert len(first[1]) == 200
    assert not set(first[0]) & set(first[1])


def test_only_development_raw_rows_are_normalized():
    raw_rows = [
        {"id": "dev-1", "title": "Dev", "context": "Evidence here.", "question": "Dev question?",
         "answers": {"text": ["Evidence"], "answer_start": [0]}},
        {"id": "heldout-1", "title": "Heldout", "context": "Heldout secret.",
         "question": "Heldout question?", "answers": {"text": ["secret"], "answer_start": [9]}},
    ]
    selected = _select_development_rows(raw_rows, "squad", ["dev-1"])
    normalized = _normalize_dev_rows(selected, "squad")
    assert [row["id"] for row in normalized] == ["dev-1"]
    assert all("heldout" not in row["question"].lower() for row in normalized)


def test_three_methods_use_one_identical_m3_seed_retrieval():
    seed = _chunk("seed", "Seed evidence.", 0, 14, next_="neighbor")
    neighbor = _chunk("neighbor", "Additional related fact.", 15, 39, prev="seed")
    vectors = {
        "Compare Alpha and Beta?": [1.0, 0.0],
        "Seed evidence.": [0.99, 0.1],
        "Additional related fact.": [0.98, 0.2],
    }
    retriever = CountingRetriever(LookupEncoder(vectors))
    chunks = [seed, neighbor]
    retriever.build(chunks)
    # One retrieval operation creates the common SearchResponse for all methods.
    search = retriever.retrieve("Compare Alpha and Beta?", top_k=1)
    before = [(hit.chunk["chunk_id"], hit.score) for hit in search.hits]
    embeddings = {chunk["chunk_id"]: retriever.vectors[i] for i, chunk in enumerate(retriever.chunks)}
    record = {
        "id": "dev-q1",
        "dataset": "squad",
        "question": "Compare Alpha and Beta?",
        "gold_evidence": [{"doc_id": "dev-doc", "start": 0, "end": 5}],
    }
    rules = {
        "single_fact": ExpansionRule(0, 0),
        "multi_concept": ExpansionRule(1, 2),
        "multi_hop": ExpansionRule(2, 4),
    }
    rows = evaluate_query_methods(
        record, search, chunks, retriever, embeddings, TinyTokenizer(), rules,
        token_budget=40, min_similarity=0.20, redundancy_threshold=0.85,
    )
    after = [(hit.chunk["chunk_id"], hit.score) for hit in search.hits]

    assert tuple(row["method"] for row in rows) == METHODS
    assert all(row["seed_chunk_ids"] == ["seed"] for row in rows)
    assert rows[0]["expanded_chunk_ids"] == []
    assert rows[1]["expanded_chunk_ids"] == ["neighbor"]
    assert rows[2]["expanded_chunk_ids"] == []  # neighbor duplicates the selected seed embedding
    assert rows[2]["neighbors_rejected_redundant"] == 1
    assert before == after
    assert retriever.retrieve_calls == 1


def test_development_config_has_fixed_threshold_and_isolated_outputs():
    config = yaml.safe_load(Path("configs/day3_dev_m4v2.yaml").read_text(encoding="utf-8"))
    assert config["development_size_per_dataset"] == 50
    assert config["heldout_size_per_dataset"] == 200
    assert config["redundancy_threshold"] == 0.85
    assert config["processed_output_dir"] == "data/processed/day3_dev_m4v2"
    assert config["results_output_dir"] == "results/day3_dev_m4v2"
