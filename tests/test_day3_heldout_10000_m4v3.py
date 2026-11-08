import hashlib
import json
import gc
import weakref
from pathlib import Path

import pytest
import yaml
import numpy as np

from query_adaptive_rag.evaluation import day3_heldout_10000_m4v3 as large
from query_adaptive_rag.evaluation import day3_heldout_m4v3 as six
from query_adaptive_rag.evaluation.large_scale_split import (
    extend_frozen_split, id_list_hash, validate_expanded_split,
)
from query_adaptive_rag.expansion.context_expander import TokenBudgetError
from query_adaptive_rag.retrieval.dense import SearchHit, SearchResponse


def _synthetic_split():
    pools = {}
    dev = {}
    heldout = {}
    for dataset, prefix in (("squad", "s"), ("hotpotqa", "h")):
        pools[dataset] = [f"{prefix}-{i}" for i in range(7000)]
        dev[dataset] = pools[dataset][:50]
        heldout[dataset] = pools[dataset][50:250]
    return pools, dev, heldout


def test_expanded_split_is_reproducible_and_preserves_frozen_ids():
    pools, dev, frozen = _synthetic_split()
    first = extend_frozen_split(pools, dev, frozen)
    second = extend_frozen_split(pools, dev, frozen)
    assert first == second
    for dataset in ("squad", "hotpotqa"):
        assert first["datasets"][dataset]["development_ids"] == dev[dataset]
        assert first["datasets"][dataset]["evaluation_ids"][:200] == frozen[dataset]
        assert first["datasets"][dataset]["evaluation_count"] == 5000
        assert first["datasets"][dataset]["additional_heldout_count"] == 4800


def test_expanded_split_has_exact_distribution_uniqueness_and_disjointness():
    pools, dev, frozen = _synthetic_split()
    manifest = extend_frozen_split(pools, dev, frozen)
    digest = validate_expanded_split(manifest, dev, frozen, frozen)
    assert len(manifest["ordered_dataset_query_ids"]) == 10_000
    assert digest == manifest["ordered_dataset_query_ids_sha256"]
    assert [sum(pair[0] == dataset for pair in manifest["ordered_dataset_query_ids"])
            for dataset in ("squad", "hotpotqa")] == [5000, 5000]


def test_split_rejects_development_overlap_or_changed_frozen_ids():
    pools, dev, frozen = _synthetic_split()
    bad_dev = {**dev, "squad": [frozen["squad"][0], *dev["squad"][1:]]}
    with pytest.raises(ValueError, match="overlap"):
        extend_frozen_split(pools, bad_dev, frozen)
    manifest = extend_frozen_split(pools, dev, frozen)
    wrong_frozen = {**frozen, "hotpotqa": ["not-the-frozen-id", *frozen["hotpotqa"][1:]]}
    with pytest.raises(ValueError, match="frozen held-out"):
        validate_expanded_split(manifest, dev, frozen, wrong_frozen)


def test_split_hash_and_id_hash_detect_manifest_tampering():
    pools, dev, frozen = _synthetic_split()
    manifest = extend_frozen_split(pools, dev, frozen)
    assert manifest["datasets"]["squad"]["evaluation_ids_sha256"] == id_list_hash(
        manifest["datasets"]["squad"]["evaluation_ids"]
    )
    manifest["ordered_dataset_query_ids"][0][1] = "tampered"
    with pytest.raises(ValueError, match="ordered dataset/query ID list"):
        validate_expanded_split(manifest, dev, frozen, frozen)


def test_config_keeps_frozen_six_method_settings_and_isolated_locations():
    config = yaml.safe_load(Path("configs/day3_heldout_10000_m4v3.yaml").read_text(encoding="utf-8"))
    baseline = yaml.safe_load(Path("configs/day3_heldout_m4v3_alias_fix.yaml").read_text(encoding="utf-8"))
    assert config["methods"] == baseline["methods"]
    for key in ("model_name", "tokenizer_name", "device", "top_k", "token_budget",
                "fixed_overlap_tokens", "semantic_similarity_threshold", "semantic_min_tokens",
                "semantic_max_tokens", "minimum_neighbor_similarity", "expansion_rules", "m4_v2", "m4_v3"):
        assert config[key] == baseline[key]
    assert config["evaluation_size_per_dataset"] == 5000
    assert config["results_output_dir"] == "results/day3_heldout_10000_m4v3"
    assert config["processed_output_dir"] == "data/processed/day3_heldout_10000_m4v3"
    assert config["embedding_cache_dir"] == "data/cache/day3_heldout_10000_m4v3"


def test_output_directory_protection_and_preflight_layout(tmp_path):
    config = {
        "results_output_dir": "results/day3_heldout_10000_m4v3",
        "processed_output_dir": "data/processed/day3_heldout_10000_m4v3",
        "embedding_cache_dir": "data/cache/day3_heldout_10000_m4v3",
    }
    results, processed, cache = large.validate_output_locations(tmp_path, config, preflight=True)
    assert results == tmp_path / "results/day3_heldout_10000_m4v3"
    assert processed == tmp_path / "data/processed/day3_heldout_10000_m4v3"
    assert cache == tmp_path / "data/cache/day3_heldout_10000_m4v3"
    with pytest.raises(ValueError, match="isolated"):
        large.validate_output_locations(tmp_path, {**config, "processed_output_dir": "data/processed/day3"}, preflight=True)
    results.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite results"):
        large.validate_output_locations(tmp_path, config, preflight=True)


def test_runtime_loader_requests_no_answer_or_evidence_columns(monkeypatch):
    observed = []

    class FakeRows:
        def __init__(self, rows):
            self.rows = rows
            self.column_names = list(rows[0]) if rows else []
        def select_columns(self, names):
            observed.append(tuple(names))
            return FakeRows([{key: row[key] for key in names} for row in self.rows])
        def __iter__(self):
            return iter(self.rows)

    raw = {
        "squad": FakeRows([{"id": "s1", "question": "Who?", "title": "T", "context": "Text",
                            "answers": {"text": ["hidden"], "answer_start": [0]}}]),
        "hotpotqa": FakeRows([{"_id": "h1", "question": "Where?", "context": [["Page", ["Text."]]],
                               "answer": "hidden", "supporting_facts": []}]),
    }
    monkeypatch.setattr(large, "_raw_dataset", lambda config, dataset: raw[dataset])
    config = {
        "datasets": {
            "squad": {"dataset_name": "squad", "split": "validation"},
            "hotpotqa": {"dataset_name": "hotpot_qa", "dataset_config": "distractor", "split": "validation"},
        }
    }
    manifest = {"ordered_dataset_query_ids": [["squad", "s1"], ["hotpotqa", "h1"]],
                "datasets": {"squad": {"evaluation_ids": ["s1"]},
                             "hotpotqa": {"evaluation_ids": ["h1"]}}}
    records = large.load_runtime_records(config, manifest)
    assert [set(record) for record in records] == [
        {"id", "dataset", "question", "documents"}, {"id", "dataset", "question", "documents"},
    ]
    assert observed == [("id", "question", "title", "context"), ("_id", "question", "context")]
    assert all(not ({"answers", "answer", "supporting_facts", "gold_evidence"} & set(record)) for record in records)


def test_m3_shared_retrieval_and_strict_budget_use_validated_selector():
    # The shared-response identity and strict budget are enforced by the same
    # selector used by the new runner; the existing six-method regression test
    # exercises that selector with all expansion implementations.
    import inspect

    source = inspect.getsource(six.select_contexts_for_query)
    assert "searches = {method: retrievers[method].retrieve(query, top_k=top_k)" in source
    assert "result[method][\"search\"] is not m3_search" in source
    assert "M3 seed identity/ranking/scores changed" in source
    assert "token_budget" in source
    assert six.METHODS == large.METHODS


def test_shared_selector_enforces_a_hard_token_budget():
    class TinyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": str(text).split()}
        def decode(self, tokens, **kwargs):
            return " ".join(tokens)

    chunk = {"chunk_id": "seed", "doc_id": "doc", "text": "one two", "start": 0, "end": 7}
    search = SearchResponse([SearchHit(chunk, 0.9, 1)], np.asarray([1.0, 0.0], dtype=np.float32), 0.1)
    with pytest.raises(TokenBudgetError, match="exceeds budget"):
        six._selection_row(six.M3, search, [chunk], "one two", 2, 1, TinyTokenizer())


def test_precomputed_sentence_encoder_returns_the_same_vectors():
    vectors = {"first": np.asarray([1.0, 0.0], dtype=np.float32),
               "second": np.asarray([0.0, 1.0], dtype=np.float32)}
    proxy = large._PrecomputedSentenceEncoder(2, vectors)
    assert proxy.get_sentence_embedding_dimension() == 2
    np.testing.assert_array_equal(proxy.encode(["first", "second"]), np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    with pytest.raises(KeyError, match="not included"):
        proxy.encode(["missing"])


def test_split_manifest_has_deterministic_hash_representation():
    pairs = [("squad", "s1"), ("hotpotqa", "h1")]
    expected = hashlib.sha256(json.dumps(pairs, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    from query_adaptive_rag.evaluation.large_scale_split import ordered_pairs_hash
    assert ordered_pairs_hash(pairs) == expected


def test_repository_split_and_preflight_manifests_are_portable_and_valid():
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/day3_heldout_10000_m4v3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_path = root / config["split_manifest"]
    split_manifest, digest = large.validate_split_file(root, config)
    preflight = json.loads(
        (root / "data/processed/day3_heldout_10000_m4v3/preflight_manifest.json")
        .read_text(encoding="utf-8")
    )

    assert split_manifest["configuration_path"] == "configs/day3_heldout_10000_m4v3.yaml"
    assert split_manifest["ordered_dataset_query_ids_sha256"] == (
        "aa8b2aa8e010d5e843838f69930cb5eb063173d804b668f14bc147dfac38b857"
    )
    assert digest == split_manifest["ordered_dataset_query_ids_sha256"]
    assert len(split_manifest["ordered_dataset_query_ids"]) == 10_000
    assert [split_manifest["datasets"][name]["evaluation_count"]
            for name in ("squad", "hotpotqa")] == [5_000, 5_000]
    assert preflight["configuration_path"] == "configs/day3_heldout_10000_m4v3.yaml"
    assert preflight["split_manifest_path"] == "data/splits/day3_heldout_10000_m4v3.json"
    assert preflight["split_manifest_sha256"] == large._sha256_file(split_path)
    assert preflight["ordered_dataset_query_ids_sha256"] == digest
    assert preflight["output_locations"] == {
        "processed": "data/processed/day3_heldout_10000_m4v3",
        "cache": "data/cache/day3_heldout_10000_m4v3",
        "results_reserved": "results/day3_heldout_10000_m4v3",
    }
    for metadata in preflight["chunk_artifacts"].values():
        assert "\\" not in metadata["relative_path"]
        assert (root / "data/processed/day3_heldout_10000_m4v3" /
                metadata["relative_path"]).is_file()


def test_portable_repo_path_uses_forward_slashes_and_rejects_external_path(tmp_path):
    root = tmp_path / "project"
    target = root / "data" / "chunks.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("", encoding="utf-8")
    assert large._portable_repo_path(target, root) == "data/chunks.jsonl"
    with pytest.raises(ValueError):
        large._portable_repo_path(tmp_path / "outside.txt", root)


class _FakeAnnotationRows:
    def __init__(self, rows, column_names=None):
        self.rows = list(rows)
        self.column_names = list(column_names) if column_names is not None else (
            list(self.rows[0]) if self.rows else []
        )

    def filter(self, predicate):
        return _FakeAnnotationRows((row for row in self.rows if predicate(row)), self.column_names)

    def select(self, indices):
        return _FakeAnnotationRows((self.rows[index] for index in indices), self.column_names)

    def __iter__(self):
        return iter(self.rows)


def _annotation_config():
    return {
        "datasets": {
            "squad": {"dataset_name": "squad", "split": "validation"},
            "hotpotqa": {
                "dataset_name": "hotpot_qa", "dataset_config": "distractor", "split": "validation",
            },
        }
    }


def _annotation_split(squad_ids=(), hotpot_ids=("h1", "h2")):
    return {
        "datasets": {
            "squad": {"evaluation_ids": list(squad_ids)},
            "hotpotqa": {"evaluation_ids": list(hotpot_ids)},
        }
    }


def _hotpot_annotation(query_id, id_field="id"):
    return {
        id_field: query_id,
        "question": f"Question {query_id}?",
        "answer": "synthetic answer",
        "context": [["Example page", ["Synthetic context sentence."]]],
        "supporting_facts": [],
    }


def _empty_squad_rows():
    return _FakeAnnotationRows([], ("id", "question", "title", "context", "answers"))


def _empty_hotpot_rows(id_field="id"):
    return _FakeAnnotationRows([], (id_field, "question", "context", "answer", "supporting_facts"))


@pytest.mark.parametrize("id_field", ["id", "_id"])
def test_hotpot_annotations_support_both_id_column_spellings(monkeypatch, id_field):
    raw = {
        "squad": _empty_squad_rows(),
        "hotpotqa": _FakeAnnotationRows([
            _hotpot_annotation("h1", id_field), _hotpot_annotation("h2", id_field),
        ], (id_field, "question", "context", "answer", "supporting_facts")),
    }
    monkeypatch.setattr(large, "_raw_dataset", lambda config, dataset: raw[dataset])

    annotations = large._load_evaluation_annotations(_annotation_config(), _annotation_split())

    assert set(annotations) == {("hotpotqa", "h1"), ("hotpotqa", "h2")}
    assert annotations[("hotpotqa", "h1")]["id"] == "h1"


def test_squad_annotation_loading_still_uses_id(monkeypatch):
    squad = _FakeAnnotationRows([
        {"id": "s1", "question": "Who?", "title": "Example", "context": "A person is named Ada.",
         "answers": {"text": ["Ada"], "answer_start": [18]}},
    ])
    raw = {"squad": squad, "hotpotqa": _empty_hotpot_rows()}
    monkeypatch.setattr(large, "_raw_dataset", lambda config, dataset: raw[dataset])

    annotations = large._load_evaluation_annotations(_annotation_config(), _annotation_split(("s1",), ()))

    assert set(annotations) == {("squad", "s1")}
    assert annotations[("squad", "s1")]["answers"] == ["Ada"]


def test_annotation_loader_reports_missing_selected_ids(monkeypatch):
    raw = {
        "squad": _empty_squad_rows(),
        "hotpotqa": _FakeAnnotationRows([_hotpot_annotation("h1", "id")]),
    }
    monkeypatch.setattr(large, "_raw_dataset", lambda config, dataset: raw[dataset])

    with pytest.raises(ValueError, match=r"source annotation IDs.*missing=\['h2'\]"):
        large._load_evaluation_annotations(_annotation_config(), _annotation_split())


def test_annotation_loader_reports_duplicate_annotation_ids(monkeypatch):
    raw = {
        "squad": _empty_squad_rows(),
        "hotpotqa": _FakeAnnotationRows([
            _hotpot_annotation("h1", "id"),
            {**_hotpot_annotation("h1", "id"), "context": []},
            _hotpot_annotation("h2", "id"),
        ], ("id", "question", "context", "answer", "supporting_facts")),
    }
    monkeypatch.setattr(large, "_raw_dataset", lambda config, dataset: raw[dataset])

    with pytest.raises(ValueError, match=r"duplicate annotation IDs.*h1"):
        large._load_evaluation_annotations(_annotation_config(), _annotation_split())


def test_annotation_mapping_is_exact_and_ignores_unselected_source_rows(monkeypatch):
    raw = {
        "squad": _empty_squad_rows(),
        "hotpotqa": _FakeAnnotationRows([
            _hotpot_annotation("h1", "id"), _hotpot_annotation("h2", "id"),
            _hotpot_annotation("not-selected", "id"),
        ], ("id", "question", "context", "answer", "supporting_facts")),
    }
    monkeypatch.setattr(large, "_raw_dataset", lambda config, dataset: raw[dataset])

    annotations = large._load_evaluation_annotations(_annotation_config(), _annotation_split())

    assert list(annotations) == [("hotpotqa", "h1"), ("hotpotqa", "h2")]


def test_gold_annotations_load_only_after_all_context_selection():
    import inspect

    source = inspect.getsource(large.run_evaluation)
    selection_call = source.index("base.select_contexts_for_query(")
    annotation_call = source.index("_iter_evaluation_annotation_batches(")
    assert selection_call < annotation_call


class _DeterministicTestEncoder:
    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            values = [sum((index + 1) * ord(char) for index, char in enumerate(str(text)))]
            vectors.append([
                float((values[0] % 97) + 1),
                float((values[0] % 193) + 1),
                float((values[0] % 389) + 1),
                float((values[0] % 769) + 1),
            ])
        return np.asarray(vectors, dtype=np.float32)


def test_sequential_method_search_spool_preserves_exact_retrieval_and_releases_index(tmp_path, monkeypatch):
    real_retriever = large.DenseRetriever
    tracked_refs = []

    class TrackingRetriever(real_retriever):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tracked_refs.append(weakref.ref(self))

    chunks_by_method = {
        large.M1: [
            {"chunk_id": "m1-a", "doc_id": "d1", "text": "alpha fact", "start": 0, "end": 10},
            {"chunk_id": "m1-b", "doc_id": "d2", "text": "beta fact", "start": 0, "end": 9},
            {"chunk_id": "m1-c", "doc_id": "d3", "text": "gamma fact", "start": 0, "end": 10},
        ],
        large.M2: [
            {"chunk_id": "m2-a", "doc_id": "d1", "text": "alpha fact", "start": 0, "end": 10},
            {"chunk_id": "m2-b", "doc_id": "d2", "text": "beta fact", "start": 0, "end": 9},
            {"chunk_id": "m2-c", "doc_id": "d3", "text": "gamma fact", "start": 0, "end": 10},
        ],
    }
    queries = [
        {"id": "q1", "dataset": "squad", "question": "alpha?"},
        {"id": "q2", "dataset": "hotpotqa", "question": "gamma?"},
    ]
    encoder = _DeterministicTestEncoder()
    expected = {}
    for method, chunks in chunks_by_method.items():
        reference = real_retriever(encoder, "test-model")
        reference.build(chunks)
        expected[method] = [reference.retrieve(row["question"], top_k=2) for row in queries]
    monkeypatch.setattr(large, "DenseRetriever", TrackingRetriever)
    processed = tmp_path / "processed"
    (processed / "indexes").mkdir(parents=True)
    (processed / "embeddings").mkdir()
    for index, method in enumerate((large.M1, large.M2)):
        spool = processed / f"{method}.jsonl"
        large._build_method_search_spool(
            method, chunks_by_method[method], queries, encoder, "test-model", 2, processed, spool,
        )
        gc.collect()
        assert tracked_refs[index]() is None
        reader = large._SearchResponseReader(spool)
        for record, reference in zip(queries, expected[method]):
            observed = reader.next_for(record)
            assert [(hit.chunk["chunk_id"], hit.rank, hit.score) for hit in observed.hits] == [
                (hit.chunk["chunk_id"], hit.rank, hit.score) for hit in reference.hits
            ]
            np.testing.assert_array_equal(observed.query_embedding, reference.query_embedding)
        reader.assert_exhausted()
        reader.close()


def test_incremental_result_writer_preserves_rows_and_metric_summaries(tmp_path):
    rows = []
    for query_index in range(4):
        for method in (large.M3, large.M4):
            rows.append({
                "query_id": f"q{query_index}", "dataset": "squad", "method": method,
                "query_text": f"question {query_index}", "seed_chunk_ids": [f"c{query_index}"],
                "classification_features": {"terms": ["a", "b"]},
                "recall_at_1": float(query_index % 2), "recall_at_3": 1.0,
                "recall_at_5": 1.0, "mrr": 1.0, "context_precision": query_index / 4.0,
                "context_token_count": 12 + query_index, "retrieval_latency_ms": 1.25,
                "expansion_latency_ms": 0.5 if method == large.M4 else 0.0,
            })
    expected_jsonl = tmp_path / "expected.jsonl"
    expected_csv = tmp_path / "expected.csv"
    large._write_jsonl(expected_jsonl, rows)
    csv_rows = [{
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    } for row in rows]
    from query_adaptive_rag.evaluation.day3 import _write_csv
    _write_csv(expected_csv, csv_rows)

    actual_jsonl = tmp_path / "actual.jsonl"
    actual_csv = tmp_path / "actual.csv"
    writer = large._StreamingResultWriter(actual_jsonl, actual_csv)
    writer.write(rows[:3])
    writer.write(rows[3:])
    writer.close()

    assert actual_jsonl.read_bytes() == expected_jsonl.read_bytes()
    assert actual_csv.read_bytes() == expected_csv.read_bytes()
    assert large._aggregate(rows) == large._aggregate([
        {key: row[key] for key in (
            "query_id", "dataset", "method", "recall_at_1", "recall_at_3", "recall_at_5",
            "mrr", "context_precision", "context_token_count", "retrieval_latency_ms", "expansion_latency_ms",
        )} for row in rows
    ])
    assert large.base.paired_bootstrap_comparisons(rows, method_a=large.M3, method_b=large.M4) == \
        large.base.paired_bootstrap_comparisons([
            {key: row[key] for key in (
                "query_id", "dataset", "method", "recall_at_1", "recall_at_3", "recall_at_5",
                "mrr", "context_precision", "context_token_count", "retrieval_latency_ms", "expansion_latency_ms",
            )} for row in rows
        ], method_a=large.M3, method_b=large.M4)


def test_selection_spool_restores_only_scoring_fields_and_preserves_counts():
    chunk = {"chunk_id": "seed", "doc_id": "doc", "start": 0, "end": 5, "text": "a fact"}
    search = SearchResponse([SearchHit(chunk, 0.75, 1)], np.asarray([1.0, 0.0]), 4.5)
    selection = {
        "search": search, "ranked_chunks": [chunk], "selected_chunks": [chunk],
        "seed_chunk_ids": ["seed"], "seed_ranks": [1], "seed_scores": {"seed": 0.75},
        "query_embedding_sha256": "abc", "context_text": "a fact", "context_token_count": 2,
        "token_budget": 20, "expanded_chunk_ids": [], "similarity_scores": {},
        "expansion_latency_ms": 0.0, "candidate_decisions": [
            {"accepted": False}, {"accepted": True},
        ], "budget_excluded_chunk_ids": [],
    }
    item = {"query_id": "q", "dataset": "squad", "query_text": "question",
            "classification": {"complexity": "single_fact", "features": {}},
            "selections": {large.M3: selection}}
    restored = large._selection_item_from_spool(large._selection_item_for_spool(item))
    output = restored["selections"][large.M3]
    assert output["seed_chunk_ids"] == ["seed"]
    assert output["ranked_chunks"] == [{key: chunk[key] for key in ("chunk_id", "doc_id", "start", "end")}]
    assert output["context_text"] == "a fact"
    assert output["search"].latency_ms == 4.5
    assert len(output["candidate_decisions"]) == 2
    assert sum(not row["accepted"] for row in output["candidate_decisions"]) == 1
