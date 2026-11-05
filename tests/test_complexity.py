import inspect

import pytest

from query_adaptive_rag.classification.complexity import classify_query


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is the capital of France?", "single_fact"),
        ("Compare Ada Lovelace and Grace Hopper", "multi_concept"),
        ("Which film did the director of The Matrix direct that stars Keanu Reeves?", "multi_hop"),
    ],
)
def test_query_complexity_classes_and_explanatory_features(query, expected):
    result = classify_query(query)
    assert result["complexity"] == expected
    assert result["features"]["reasons"]


def test_classifier_interface_cannot_receive_gold_evidence():
    record = {
        "question": "What is the capital of France?",
        "gold_evidence": [{"doc_id": "secret", "start": 1, "end": 9}],
        "answers": ["also secret"],
        "complexity": "multi_hop",
    }
    result = classify_query(record["question"])
    assert result["complexity"] == "single_fact"
    assert list(inspect.signature(classify_query).parameters) == ["query"]
    with pytest.raises(TypeError):
        classify_query(record["question"], gold_evidence=record["gold_evidence"])


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        classify_query("   ")
