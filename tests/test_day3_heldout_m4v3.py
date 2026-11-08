import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from query_adaptive_rag.evaluation import day3_heldout_m4v3 as heldout
from query_adaptive_rag.expansion.context_expander import ExpansionRule, TokenBudgetError
from query_adaptive_rag.expansion.context_expander_v3 import ExpansionV3Config
from query_adaptive_rag.retrieval.dense import SearchHit, SearchResponse


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": str(text).split()}

    def decode(self, token_ids, **kwargs):
        return " ".join(token_ids)


def _split(ids_by_dataset, development_ids=None):
    development_ids = development_ids or {dataset: [] for dataset in ids_by_dataset}
    return {dataset: {"evaluation_ids": ids, "development_ids": development_ids[dataset]}
            for dataset, ids in ids_by_dataset.items()}


def _manifest(ids_by_dataset, key):
    return {"dataset_splits": {dataset: {key: ids} for dataset, ids in ids_by_dataset.items()}}


def _split_reference(evaluation_ids, development_ids):
    return {"dataset_splits": {
        dataset: {"evaluation_ids": evaluation_ids[dataset], "development_ids": development_ids[dataset]}
        for dataset in evaluation_ids
    }}


def test_saved_heldout_ids_match_original_corrected_and_development_manifests():
    config = yaml.safe_load(Path("configs/day3_heldout_m4v3_alias_fix.yaml").read_text(encoding="utf-8"))
    original = json.loads(Path("results/day3/experiment_manifest.json").read_text(encoding="utf-8"))
    corrected = json.loads(Path("results/day3_corrected/experiment_manifest.json").read_text(encoding="utf-8"))
    development = json.loads(Path("results/day3_dev_m4v3_fact_alias_fix/experiment_manifest.json").read_text(encoding="utf-8"))

    pairs, digest = heldout.validate_heldout_split(
        original["dataset_splits"], original, corrected, development,
        config["expected_heldout_query_ids_sha256"],
    )
    assert len(pairs) == 400
    assert digest == heldout.EXPECTED_SPLIT_HASH
    assert [sum(dataset == name for dataset, _ in pairs) for name in ("squad", "hotpotqa")] == [200, 200]


def test_split_validator_rejects_id_order_hash_and_development_overlap():
    squads = [f"s-{i}" for i in range(200)]
    hotpot = [f"h-{i}" for i in range(200)]
    ids = {"squad": squads, "hotpotqa": hotpot}
    dev_ids = {"squad": [f"sd-{i}" for i in range(50)], "hotpotqa": [f"hd-{i}" for i in range(50)]}
    split = _split(ids, dev_ids)
    development = _manifest(dev_ids, "development_ids")
    original = _split_reference(ids, dev_ids)
    corrected = _split_reference(ids, dev_ids)
    pairs = [(ds, qid) for ds in ("squad", "hotpotqa") for qid in ids[ds]]
    digest = heldout._ordered_id_hash(pairs)
    assert len(heldout.validate_heldout_split(split, original, corrected, development, digest)[0]) == 400

    with pytest.raises(ValueError, match="hash mismatch"):
        heldout.validate_heldout_split(split, original, corrected, development, "0" * 64)
    overlapping_ids = {"squad": [squads[0], *dev_ids["squad"][1:]], "hotpotqa": dev_ids["hotpotqa"]}
    overlapping = _manifest(overlapping_ids, "development_ids")
    with pytest.raises(ValueError, match="overlap"):
        heldout.validate_heldout_split(
            _split(ids, overlapping_ids), _split_reference(ids, overlapping_ids),
            _split_reference(ids, overlapping_ids), overlapping, digest,
        )


def test_output_directory_safety_requires_fresh_isolated_root(tmp_path):
    results = tmp_path / "results" / "day3_heldout_m4v3_alias_fix"
    heldout.validate_output_paths(tmp_path, results, results / "processed", results / "cache")

    results.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        heldout.validate_output_paths(tmp_path, results, results / "processed", results / "cache")
    with pytest.raises(ValueError, match="isolated destination"):
        heldout.validate_output_paths(tmp_path, tmp_path / "results" / "day3", tmp_path / "results" / "day3" / "processed",
                                      tmp_path / "results" / "day3" / "cache")


def test_frozen_settings_match_corrected_protocol_and_development_expanders():
    root = Path.cwd()
    config = yaml.safe_load((root / "configs/day3_heldout_m4v3_alias_fix.yaml").read_text(encoding="utf-8"))
    corrected = json.loads((root / "results/day3_corrected/experiment_manifest.json").read_text(encoding="utf-8"))
    v2 = yaml.safe_load((root / "configs/day3_dev_m4v2.yaml").read_text(encoding="utf-8"))
    v3 = yaml.safe_load((root / "configs/day3_dev_m4v3_fact.yaml").read_text(encoding="utf-8"))
    heldout.validate_frozen_settings(config, corrected, v2, v3)

    changed = dict(config)
    changed["m4_v2"] = {"redundancy_threshold": 0.7}
    with pytest.raises(ValueError, match="redundancy threshold"):
        heldout.validate_frozen_settings(changed, corrected, v2, v3)


def test_integrity_hashes_record_config_and_source_files(tmp_path):
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "runner.py"
    config = tmp_path / "run.yaml"
    source.write_text("runner", encoding="utf-8")
    config.write_text("run_id: example\n", encoding="utf-8")
    record = heldout.integrity_hashes(tmp_path, config, source_paths=("src/runner.py",))
    assert record["configuration_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert record["source_sha256"]["src/runner.py"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["source_sha256"]["config"] == record["configuration_sha256"]


def test_selection_executes_six_methods_using_one_shared_m3_search(monkeypatch):
    tokenizer = TinyTokenizer()
    chunks = {
        heldout.M1: [{"chunk_id": "m1", "doc_id": "d", "text": "m1 text", "start": 0, "end": 7}],
        heldout.M2: [{"chunk_id": "m2", "doc_id": "d", "text": "m2 text", "start": 0, "end": 7}],
    }
    seed = {"chunk_id": "seed", "doc_id": "d", "text": "seed text", "start": 0, "end": 9,
            "prev_chunk_id": None, "next_chunk_id": "neighbor", "method": "semantic"}
    neighbor = {"chunk_id": "neighbor", "doc_id": "d", "text": "neighbor text", "start": 10, "end": 23,
                "prev_chunk_id": "seed", "next_chunk_id": None, "method": "semantic"}
    chunks[heldout.M3] = [seed, neighbor]

    class FakeRetriever:
        def __init__(self, chunk):
            self.chunk = chunk
            self.calls = 0
            self.response = SearchResponse([SearchHit(chunk, 0.9, 1)], np.asarray([1.0, 0.0], dtype=np.float32), 1.0)
        def retrieve(self, query, top_k=5):
            self.calls += 1
            return self.response
        def score(self, query_embedding, chunk_id):
            return 0.9

    retrievers = {
        heldout.M1: FakeRetriever(chunks[heldout.M1][0]),
        heldout.M2: FakeRetriever(chunks[heldout.M2][0]),
        heldout.M3: FakeRetriever(seed),
    }
    retrievers[heldout.M3].chunks = [seed, neighbor]
    retrievers[heldout.M3].vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    captured = {}

    def result(search, seed_ids, text="seed text"):
        return SimpleNamespace(seed_chunk_ids=seed_ids, seed_ranks=[hit.rank for hit in search.hits],
                               expanded_chunk_ids=[], similarity_scores={}, redundancy_scores={},
                               context_text=text, context_token_count=2, token_budget=50, latency_ms=1.0,
                               budget_rejected_chunk_ids=[], deduplicated_chunk_ids=[],
                               similarity_rejected_chunk_ids=[], redundancy_rejected_chunk_ids=[],
                               candidate_decisions=[])

    def fake_m4(**kwargs):
        captured[heldout.M4] = kwargs["search"]
        return result(kwargs["search"], ["seed"])

    def fake_m4v2(**kwargs):
        captured[heldout.M4V2] = kwargs["search"]
        return result(kwargs["search"], ["seed"])

    def fake_m4v3(search, *args, **kwargs):
        captured[heldout.M4V3] = search
        return result(search, ["seed"])

    monkeypatch.setattr(heldout, "expand_context", fake_m4)
    monkeypatch.setattr(heldout, "expand_context_v2", fake_m4v2)
    monkeypatch.setattr(heldout, "expand_context_v3", fake_m4v3)
    shared_chunk_embeddings = {
        "seed": np.asarray([1.0, 0.0], dtype=np.float32),
        "neighbor": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    monkeypatch.setattr(
        heldout, "chunk_embedding_map_from_retriever",
        lambda _: pytest.fail("the caller-provided M3 chunk map should be reused"),
    )
    selected = heldout.select_contexts_for_query(
        "query text", "single_fact", chunks, retrievers, tokenizer,
        {"single_fact": ExpansionRule(0, 0)}, 50, 0.2, 0.85, ExpansionV3Config(), 5,
        chunk_embeddings=shared_chunk_embeddings,
    )
    m3_search = retrievers[heldout.M3].response
    assert set(selected) == set(heldout.METHODS)
    assert retrievers[heldout.M3].calls == 1
    assert all(captured[method] is m3_search for method in heldout.EXPANSION_METHODS)
    assert all(selected[method]["seed_chunk_ids"] == ["seed"] for method in heldout.EXPANSION_METHODS)
    assert all(selected[method]["seed_ranks"] == [1] for method in heldout.EXPANSION_METHODS)
    assert len({selected[method]["query_embedding_sha256"] for method in heldout.EXPANSION_METHODS}) == 1


def test_selection_api_has_no_gold_or_answer_inputs_and_score_is_separate():
    import inspect
    params = set(inspect.signature(heldout.select_contexts_for_query).parameters)
    assert not params & {"answers", "gold_answers", "gold_evidence", "evidence", "labels", "metrics"}
    score_params = set(inspect.signature(heldout.score_selected_contexts).parameters)
    assert score_params == {"pending", "records", "token_budget"}


def test_gold_evidence_is_not_given_to_selection_and_metrics_run_after_all_selections(monkeypatch):
    chunk = {"chunk_id": "s", "doc_id": "d", "text": "fact about town", "start": 0, "end": 15}
    search = SearchResponse([SearchHit(chunk, 0.9, 1)], np.asarray([1.0, 0.0]), 0.1)
    records = [
        {"id": f"q{i}", "dataset": "squad", "question": f"query {i}", "documents": [{"doc_id": "d", "text": chunk["text"]}],
         "gold_evidence": [{"doc_id": "d", "start": 0, "end": 4}], "answers": ["hidden answer"]}
        for i in range(2)
    ]
    events = []
    def fake_select(query, *args):
        events.append(("selection", query))
        selected = {}
        for method in heldout.METHODS:
            selected[method] = {
                "search": search, "ranked_chunks": [chunk], "selected_chunks": [chunk],
                "seed_chunk_ids": ["s"], "seed_ranks": [1], "seed_scores": {"s": 0.9},
                "query_embedding_sha256": "abc", "context_text": chunk["text"],
                "context_token_count": 3, "token_budget": 100, "expanded_chunk_ids": [],
                "similarity_scores": {}, "expansion_latency_ms": 0.0, "candidate_decisions": [],
                "budget_excluded_chunk_ids": [],
            }
        return selected
    monkeypatch.setattr(heldout, "classify_query", lambda query: {"complexity": "single_fact", "features": {}})
    monkeypatch.setattr(heldout, "select_contexts_for_query", fake_select)
    runtime_records = heldout.runtime_query_views(records)
    assert all("gold_evidence" not in row and "answers" not in row for row in runtime_records)
    pending, _ = heldout.select_all_contexts(
        runtime_records, {}, {}, TinyTokenizer(), {}, 100, 0.2, 0.85, ExpansionV3Config(), 5,
    )
    assert events == [("selection", "query 0"), ("selection", "query 1")]

    real_metrics = heldout.calculate_metrics
    def traced_metrics(*args):
        events.append(("metric", None))
        return real_metrics(*args)
    monkeypatch.setattr(heldout, "calculate_metrics", traced_metrics)
    output = heldout.score_selected_contexts(pending, records, 100)
    assert len(output) == 2 * len(heldout.METHODS)
    assert [kind for kind, _ in events] == ["selection", "selection"] + ["metric"] * 12


def test_selection_row_rejects_context_over_budget():
    chunk = {"chunk_id": "seed", "doc_id": "d", "text": "one two", "start": 0, "end": 7}
    search = SearchResponse([SearchHit(chunk, 0.9, 1)], np.asarray([1.0, 0.0]), 0.1)
    with pytest.raises(TokenBudgetError, match="exceeds budget"):
        heldout._selection_row(heldout.M3, search, [chunk], "one two", 2, 1, TinyTokenizer())
