import logging

from query_adaptive_rag.config import load_config, set_random_seed
from query_adaptive_rag.data.loaders import _hotpot_records, _squad_records


def test_squad_normalization():
    records = _squad_records([{
        "id": "q1", "title": "Title", "context": "Answer is here.", "question": "What?",
        "answers": {"text": ["Answer"], "answer_start": [0]},
    }])
    doc_id = records[0]["documents"][0]["doc_id"]
    assert doc_id.startswith("squad:Title:")
    assert records[0]["gold_evidence"] == [{"doc_id": doc_id, "start": 0, "end": 6}]
    assert records[0]["complexity"] == "single-fact"


def test_squad_original_duplicate_answer_annotations_are_preserved():
    records = _squad_records([{
        "id": "q-duplicate", "title": "Title", "context": "Answer here.", "question": "What?",
        "answers": {"text": ["Answer", "Answer"], "answer_start": [0, 0]},
    }])
    assert records[0]["answers"] == ["Answer", "Answer"]
    assert records[0]["gold_evidence"] == [
        {"doc_id": records[0]["documents"][0]["doc_id"], "start": 0, "end": 6},
        {"doc_id": records[0]["documents"][0]["doc_id"], "start": 0, "end": 6},
    ]


def test_hotpot_normalization_supporting_sentence_offset():
    rows = [{
        "_id": "h1", "question": "Why?", "answer": "Because", "context": [["Page", ["First sentence.", "Second one."]]],
        "supporting_facts": [["Page", 1]],
    }]
    record = _hotpot_records(rows)[0]
    assert record["documents"][0]["text"] == "First sentence. Second one."
    assert record["gold_evidence"][0]["doc_id"] == record["documents"][0]["doc_id"]
    assert record["gold_evidence"][0]["start"] == 16
    assert record["gold_evidence"][0]["end"] == 27
    assert record["complexity"] == "multi-hop"


def test_hotpot_huggingface_columnar_schema_preserves_sentences_and_facts():
    record = _hotpot_records([{
        "id": "h-columnar",
        "question": "What connects these pages?",
        "answer": "A shared director",
        "context": {
            "title": ["Film", "Director"],
            "sentences": [["Intro.", "Evidence."], ["Profile."]],
        },
        "supporting_facts": {
            "title": ["Film", "Director"],
            "sent_id": [1, 0],
        },
    }])[0]

    first_doc, second_doc = record["documents"]
    assert record["id"] == "h-columnar"
    assert first_doc["title"] == "Film"
    assert first_doc["sentences"] == ["Intro.", "Evidence."]
    assert first_doc["text"] == "Intro. Evidence."
    assert second_doc["sentences"] == ["Profile."]
    assert record["supporting_facts"] == [
        {"title": "Film", "sentence_index": 1},
        {"title": "Director", "sentence_index": 0},
    ]
    assert record["gold_evidence"] == [
        {"doc_id": first_doc["doc_id"], "start": 7, "end": 16},
        {"doc_id": second_doc["doc_id"], "start": 0, "end": 8},
    ]


def test_hotpot_malformed_rows_and_facts_are_skipped_safely(caplog):
    with caplog.at_level(logging.WARNING):
        records = _hotpot_records([
            {"id": "missing-context", "question": "Why?", "answer": "Unknown"},
            {
                "id": "bad-facts",
                "question": "Why?",
                "answer": "Known",
                "context": {"title": ["Page"], "sentences": [["Only sentence."]]},
                "supporting_facts": {"title": ["Page"], "sent_id": ["not-an-index"]},
            },
        ])

    assert len(records) == 1
    assert records[0]["id"] == "bad-facts"
    assert records[0]["supporting_facts"] == []
    assert records[0]["gold_evidence"] == []
    assert "without valid context documents" in caplog.text
    assert "invalid sentence index" in caplog.text


def test_config_resolves_paths_and_seed():
    config = load_config("configs/day1.yaml")
    assert config.fixed_chunk_sizes == (256, 512)
    assert config.fixed_overlap_tokens == 32
    assert config.output_dir.is_absolute()
    set_random_seed(config.seed)
