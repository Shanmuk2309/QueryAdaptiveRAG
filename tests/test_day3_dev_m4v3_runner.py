import json
from pathlib import Path

import pytest

from query_adaptive_rag.evaluation.day3_dev_m4v3 import _load_reused_semantic_chunks


def _write_chunks(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _chunk(chunk_id: str, doc_id: str, text: str, start: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "method": "semantic",
        "text": text,
        "start": start,
        "end": start + len(text),
    }


def test_reused_chunks_require_expected_hash_before_json_parsing(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("not valid json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file hash mismatch"):
        _load_reused_semantic_chunks(path, [{"doc_id": "d1", "text": "source"}], "wrong-hash")


def test_reused_chunks_must_match_development_corpus_and_source_offsets(tmp_path):
    corpus = [
        {"doc_id": "d1", "text": "first document"},
        {"doc_id": "d2", "text": "second document"},
    ]
    chunks = [
        _chunk("d1::c0", "d1", "first", 0),
        _chunk("d2::c0", "d2", "second", 0),
    ]
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path, chunks)

    assert _load_reused_semantic_chunks(path, corpus) == chunks

    wrong_document = [*chunks, _chunk("d3::c0", "d3", "other", 0)]
    _write_chunks(path, wrong_document)
    with pytest.raises(ValueError, match="unexpected document"):
        _load_reused_semantic_chunks(path, corpus)

    wrong_offsets = [chunks[0], _chunk("d2::c0", "d2", "ond", 0)]
    _write_chunks(path, wrong_offsets)
    with pytest.raises(ValueError, match="text/offset mismatch"):
        _load_reused_semantic_chunks(path, corpus)


def test_reused_chunks_must_cover_each_development_document_once_by_id(tmp_path):
    corpus = [
        {"doc_id": "d1", "text": "first document"},
        {"doc_id": "d2", "text": "second document"},
    ]
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path, [_chunk("d1::c0", "d1", "first", 0)])
    with pytest.raises(ValueError, match="do not cover exactly"):
        _load_reused_semantic_chunks(path, corpus)

    duplicate = [
        _chunk("same-id", "d1", "first", 0),
        _chunk("same-id", "d2", "second", 0),
    ]
    _write_chunks(path, duplicate)
    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        _load_reused_semantic_chunks(path, corpus)
