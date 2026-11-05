import pytest

from query_adaptive_rag.evaluation.evidence import deduplicate_evidence, evidence_denominator


def test_duplicate_gold_spans_do_not_inflate_retrieval_recall_denominator():
    gold = [
        {"doc_id": "doc-a", "start": 10, "end": 20},
        {"doc_id": "doc-a", "start": 10, "end": 20},
        {"document_id": "doc-b", "start": 30, "end": 40},
    ]
    unique_gold = deduplicate_evidence(gold)
    retrieved_keys = {("doc-a", 10, 20)}
    matched = sum((item.get("document_id", item.get("doc_id")), item["start"], item["end"]) in retrieved_keys
                  for item in unique_gold)

    assert evidence_denominator(gold) == 2
    assert len(gold) == 3  # source annotations are still intact
    assert matched / evidence_denominator(gold) == 0.5


def test_evidence_dedup_preserves_first_annotation_and_order():
    spans = [
        {"doc_id": "doc", "start": 0, "end": 2, "source": "first"},
        {"doc_id": "doc", "start": 3, "end": 4, "source": "second"},
        {"doc_id": "doc", "start": 0, "end": 2, "source": "duplicate"},
    ]
    assert [x["source"] for x in deduplicate_evidence(spans)] == ["first", "second"]


def test_evidence_dedup_rejects_malformed_spans():
    with pytest.raises(ValueError):
        deduplicate_evidence([{"doc_id": "doc", "start": 0}])
