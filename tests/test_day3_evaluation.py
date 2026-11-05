import pytest
import yaml
from pathlib import Path

from query_adaptive_rag.evaluation.metrics import calculate_metrics, context_precision, recall_at_k, reciprocal_rank
from query_adaptive_rag.evaluation.context import context_source_chunks
from query_adaptive_rag.evaluation.protocol import METHODS, method_protocol
from query_adaptive_rag.evaluation.splits import deterministic_split
from query_adaptive_rag.classification.complexity import classify_query


def test_evidence_metrics_deduplicate_gold_spans_and_use_overlap():
    evidence = [
        {"doc_id": "doc-a", "start": 10, "end": 20},
        {"doc_id": "doc-a", "start": 10, "end": 20},  # duplicate annotation
        {"document_id": "doc-b", "start": 0, "end": 5},
    ]
    ranked = [
        {"chunk_id": "irrelevant", "doc_id": "doc-a", "start": 0, "end": 8, "text": "none"},
        {"chunk_id": "hit-a", "doc_id": "doc-a", "start": 18, "end": 24, "text": "answer"},
        {"chunk_id": "hit-b", "doc_id": "doc-b", "start": 0, "end": 5, "text": "second"},
    ]
    metrics = calculate_metrics(ranked, ranked, evidence)
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_3"] == 1.0  # 2 unique evidence spans, both covered
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["context_precision"] == pytest.approx(2 / 3)
    assert recall_at_k(ranked, evidence, 3) != 2 / 3
    assert reciprocal_rank(ranked, evidence) == 0.5


def test_duplicate_context_chunk_id_does_not_inflate_context_precision():
    evidence = [{"doc_id": "doc", "start": 0, "end": 5}]
    hit = {"chunk_id": "hit", "doc_id": "doc", "start": 0, "end": 5, "text": "answer"}
    irrelevant = {"chunk_id": "other", "doc_id": "doc", "start": 10, "end": 15, "text": "other"}
    assert context_precision([hit, hit, irrelevant], evidence) == 0.5


def test_context_precision_does_not_credit_text_removed_as_overlap():
    first = {"chunk_id": "first", "doc_id": "doc", "start": 0, "end": 12, "text": "shared words"}
    later = {"chunk_id": "later", "doc_id": "doc", "start": 7, "end": 18, "text": "words tail."}
    evidence_in_duplicate_prefix = [{"doc_id": "doc", "start": 8, "end": 11}]
    effective = context_source_chunks([first, later])
    assert effective[1]["context_start"] == 12
    assert context_precision(effective, evidence_in_duplicate_prefix) == 0.5


def test_heldout_split_is_reproducible_disjoint_and_complete():
    records = [{"id": str(i)} for i in range(30)]
    dev1, eval1 = deterministic_split(records, 5, 10, 123)
    dev2, eval2 = deterministic_split(records, 5, 10, 123)
    assert [r["id"] for r in dev1] == [r["id"] for r in dev2]
    assert [r["id"] for r in eval1] == [r["id"] for r in eval2]
    assert not ({r["id"] for r in dev1} & {r["id"] for r in eval1})
    assert len(dev1) + len(eval1) == 15


def test_method_protocol_uses_shared_model_topk_budget_and_m3_seeds_for_m4():
    protocol = method_protocol("all-MiniLM-L6-v2", top_k=5, token_budget=3072, fixed_overlap_tokens=32,
                               embedding_max_sequence_length=512)
    assert tuple(protocol) == METHODS
    assert {item["embedding_model"] for item in protocol.values()} == {"all-MiniLM-L6-v2"}
    assert {item["top_k"] for item in protocol.values()} == {5}
    assert {item["token_budget"] for item in protocol.values()} == {3072}
    assert {item["embedding_max_sequence_length"] for item in protocol.values()} == {512}
    assert protocol["M1_fixed_256"]["chunk_size"] == 256
    assert protocol["M2_fixed_512"]["chunk_size"] == 512
    assert protocol["M3_semantic"]["chunking"] == "semantic"
    assert protocol["M4_adaptive"]["seed_method"] == "M3_semantic"


def test_evaluation_labels_and_evidence_do_not_change_runtime_classification():
    query = "What is the capital of France?"
    result = classify_query(query)
    modified_evaluation_record = {
        "question": query,
        "complexity": "multi_hop",
        "gold_evidence": [{"doc_id": "other", "start": 1, "end": 5}],
    }
    assert classify_query(modified_evaluation_record["question"]) == result


def test_day3_config_has_balanced_heldout_design_and_shared_budget():
    config = yaml.safe_load(Path("configs/day3.yaml").read_text(encoding="utf-8"))
    assert config["candidate_pool_size_per_dataset"] == (
        config["development_size_per_dataset"] + config["evaluation_size_per_dataset"]
    )
    assert config["evaluation_size_per_dataset"] == 200
    assert config["top_k"] == 5
    assert config["token_budget"] > 5 * config["semantic_max_tokens"]
    assert config["embedding_max_sequence_length"] == 512
    assert config["datasets"]["squad"]["split"] == config["datasets"]["hotpotqa"]["split"]
