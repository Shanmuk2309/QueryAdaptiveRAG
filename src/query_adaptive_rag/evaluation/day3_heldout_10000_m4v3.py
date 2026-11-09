"""Isolated preparation and six-method runner for a 10,000-query held-out set.

Split creation reads only dataset IDs. Preflight/runtime views omit answer and
evidence columns. Gold annotations are loaded only after all contexts are
selected, for metric calculation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import gc
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import yaml

from ..chunking.fixed_size import fixed_size_chunks
from ..chunking.semantic import semantic_chunks
from ..chunking.text_utils import OffsetTokenizer, segment_sentences
from ..data.loaders import _document_id, _hotpot_context_entries, _hotpot_records, _squad_records
from ..evaluation.token_limits import audit_chunk_embedding_lengths, encoder_content_token_limit
from ..retrieval.dense import DenseRetriever, SearchHit, SearchResponse
from . import day3_heldout_m4v3 as base
from .day3 import _aggregate, _dependency_versions, _unique_corpus, _write_csv, _write_jsonl
from .large_scale_split import (
    DATASETS, DEV_PER_DATASET, TARGET_PER_DATASET, extend_frozen_split,
    id_list_hash, validate_expanded_split,
)


METHODS = base.METHODS
M1, M2, M3, M4, M4V2, M4V3 = METHODS
SPLIT_SCHEMA_VERSION = 1
SOURCE_HASH_PATHS = (
    "src/query_adaptive_rag/evaluation/day3_heldout_10000_m4v3.py",
    "src/query_adaptive_rag/evaluation/large_scale_split.py",
    "src/query_adaptive_rag/evaluation/day3_heldout_m4v3.py",
    "src/query_adaptive_rag/evaluation/day3.py",
    "src/query_adaptive_rag/evaluation/splits.py",
    "src/query_adaptive_rag/evaluation/statistics.py",
    "src/query_adaptive_rag/evaluation/metrics.py",
    "src/query_adaptive_rag/evaluation/evidence.py",
    "src/query_adaptive_rag/evaluation/context.py",
    "src/query_adaptive_rag/evaluation/token_limits.py",
    "src/query_adaptive_rag/chunking/fixed_size.py",
    "src/query_adaptive_rag/chunking/semantic.py",
    "src/query_adaptive_rag/classification/complexity.py",
    "src/query_adaptive_rag/expansion/context_expander.py",
    "src/query_adaptive_rag/expansion/context_expander_v2.py",
    "src/query_adaptive_rag/expansion/context_expander_v3.py",
    "src/query_adaptive_rag/expansion/query_facets.py",
    "src/query_adaptive_rag/retrieval/dense.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _root_and_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    return root, path, config


def _portable_repo_path(path: str | Path, root: Path) -> str:
    """Represent a repository-local path independently of the machine root."""
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _dataset_splits(config: Mapping[str, Any]):
    from datasets import load_dataset

    result = {}
    for dataset in DATASETS:
        spec = config["datasets"][dataset]
        if dataset == "squad":
            rows = load_dataset(spec["dataset_name"], split=spec["split"])
            id_column, load_seed = "id", int(config["seed"])
        else:
            rows = load_dataset(spec["dataset_name"], spec["dataset_config"], split=spec["split"])
            id_column = "_id" if "_id" in rows.column_names else "id"
            load_seed = int(config["seed"]) + 1
        # Only the ID column is retained or read during split generation.
        ids = rows.select_columns([id_column]).shuffle(seed=load_seed)[id_column]
        result[dataset] = [str(value) for value in ids]
    return result


def _load_reference_manifests(root: Path, config: Mapping[str, Any]):
    def read(name: str) -> dict[str, Any]:
        return json.loads((root / config[name]).read_text(encoding="utf-8"))

    original, corrected, development = (
        read("original_reference_manifest"), read("reference_manifest"),
        read("development_reference_manifest"),
    )
    development_ids = {}
    original_ids = {}
    corrected_ids = {}
    for dataset in DATASETS:
        original_split = original["dataset_splits"][dataset]
        corrected_split = corrected["dataset_splits"][dataset]
        dev_split = development["dataset_splits"][dataset]
        dev = [str(value) for value in dev_split["development_ids"]]
        old_eval = [str(value) for value in original_split["evaluation_ids"]]
        corrected_eval = [str(value) for value in corrected_split["evaluation_ids"]]
        if old_eval != corrected_eval:
            raise ValueError(f"{dataset} original/corrected held-out IDs or order differ")
        if dev != [str(value) for value in original_split["development_ids"]] or dev != [
            str(value) for value in corrected_split["development_ids"]
        ]:
            raise ValueError(f"{dataset} development IDs differ across references")
        development_ids[dataset] = dev
        original_ids[dataset] = old_eval
        corrected_ids[dataset] = corrected_eval
    return original, corrected, development, development_ids, original_ids, corrected_ids


def create_split(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Create an ID-only split manifest while preserving all frozen assignments."""
    root, config_file, config = _root_and_config(config_path)
    split_path = root / config["split_manifest"]
    processed_dir = root / config["processed_output_dir"]
    cache_dir = root / config["embedding_cache_dir"]
    results_dir = root / config["results_output_dir"]
    if split_path.exists():
        raise FileExistsError(f"refusing to overwrite split manifest: {split_path}")
    if any(path.exists() for path in (processed_dir, cache_dir, results_dir)):
        raise FileExistsError("refusing split preparation because an isolated output path already exists")

    original, corrected, development, dev_ids, old_ids, corrected_ids = _load_reference_manifests(root, config)
    if old_ids != corrected_ids:
        raise ValueError("frozen held-out IDs differ between reference manifests")
    source_ids = _dataset_splits(config)
    generated = extend_frozen_split(
        source_ids, dev_ids, old_ids, int(config["evaluation_size_per_dataset"]),
    )
    generated["run_id"] = str(config["run_id"])
    generated["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    generated["source_datasets"] = {
        dataset: {
            "dataset_name": config["datasets"][dataset]["dataset_name"],
            "dataset_config": config["datasets"][dataset].get("dataset_config"),
            "source_split": config["datasets"][dataset]["split"],
            "source_record_count": len(source_ids[dataset]),
            "load_seed": generated["datasets"][dataset]["load_seed"],
        }
        for dataset in DATASETS
    }
    generated["frozen_reference_ids_sha256"] = {
        dataset: {
            "development": id_list_hash(dev_ids[dataset]),
            "heldout": id_list_hash(old_ids[dataset]),
        }
        for dataset in DATASETS
    }
    digest = validate_expanded_split(generated, dev_ids, old_ids, corrected_ids)
    generated["ordered_dataset_query_ids_sha256"] = digest
    generated["configuration_path"] = _portable_repo_path(config_file, root)
    generated["reference_manifest_hashes"] = {
        key: _sha256_file(root / config[key])
        for key in ("original_reference_manifest", "reference_manifest", "development_reference_manifest")
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return split_path, generated


def validate_split_file(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    manifest = json.loads((root / config["split_manifest"]).read_text(encoding="utf-8"))
    _, _, _, dev_ids, old_ids, corrected_ids = _load_reference_manifests(root, config)
    digest = validate_expanded_split(
        manifest, dev_ids, old_ids, corrected_ids,
        int(config["evaluation_size_per_dataset"]),
    )
    expected = str(config.get("expected_heldout_query_ids_sha256", ""))
    if not expected or digest != expected:
        raise ValueError("expanded split hash differs from the frozen configuration")
    if manifest.get("total_evaluation_count") != 10_000:
        raise ValueError("split manifest must contain exactly 10,000 held-out query IDs")
    return manifest, digest


def validate_output_locations(root: Path, config: Mapping[str, Any], *, preflight: bool) -> tuple[Path, Path, Path]:
    results = (root / config["results_output_dir"]).resolve()
    processed = (root / config["processed_output_dir"]).resolve()
    cache = (root / config["embedding_cache_dir"]).resolve()
    expected = (root / "results" / "day3_heldout_10000_m4v3").resolve()
    if results != expected:
        raise ValueError(f"results must stay in the isolated directory {expected}")
    if processed != (root / "data" / "processed" / "day3_heldout_10000_m4v3").resolve():
        raise ValueError("processed data must use the isolated day3_heldout_10000_m4v3 directory")
    if cache != (root / "data" / "cache" / "day3_heldout_10000_m4v3").resolve():
        raise ValueError("embedding cache must use the isolated day3_heldout_10000_m4v3 directory")
    for target in (processed, cache):
        if root.resolve() not in target.parents:
            raise ValueError(f"output path escapes the repository: {target}")
    protected = [
        root / "results" / name for name in (
            "day3", "day3_corrected", "day3_dev_m4v2", "day3_dev_m4v3",
            "day3_dev_m4v3_fact", "day3_dev_m4v3_fact_alias_fix",
            "day3_heldout_m4v3_alias_fix",
        )
    ] + [
        root / "data" / parent / name for parent in ("processed", "cache") for name in (
            "day3", "day3_corrected", "day3_dev_m4v2", "day3_dev_m4v3",
            "day3_dev_m4v3_fact", "day3_dev_m4v3_fact_alias_fix",
        )
    ]
    for protected_path in protected:
        resolved = protected_path.resolve()
        if any(path == resolved or path in resolved.parents or resolved in path.parents
               for path in (results, processed, cache)):
            raise ValueError(f"new run path overlaps protected experiment output: {resolved}")
    if results.exists():
        raise FileExistsError(f"refusing to overwrite results directory: {results}")
    preflight_manifest = processed / "preflight_manifest.json"
    if preflight:
        if processed.exists():
            raise FileExistsError("refusing to overwrite pre-existing processed preflight artifacts")
        if cache.exists():
            cached_files = list(cache.iterdir())
            if any(not item.is_file() or not item.name.startswith("sentences-") or item.suffix != ".npy"
                   for item in cached_files):
                raise FileExistsError("existing isolated cache contains unexpected files")
    else:
        if not processed.is_dir() or not cache.is_dir() or not preflight_manifest.is_file():
            raise FileNotFoundError("matching full-corpus preflight artifacts are required before evaluation")
    return results, processed, cache


def _raw_dataset(config: Mapping[str, Any], dataset: str):
    from datasets import load_dataset

    spec = config["datasets"][dataset]
    if dataset == "squad":
        return load_dataset(spec["dataset_name"], split=spec["split"])
    return load_dataset(spec["dataset_name"], spec["dataset_config"], split=spec["split"])


def _dataset_id_column(dataset: str, raw: Any) -> str:
    """Choose the dataset's actual ID column, matching runtime loading."""
    columns = set(getattr(raw, "column_names", ()))
    if dataset == "squad":
        if "id" not in columns:
            raise ValueError(f"SQuAD dataset is missing required ID column 'id'; available columns: {sorted(columns)}")
        return "id"
    if dataset == "hotpotqa":
        # The upstream data has appeared with either spelling. Prefer `_id`
        # when both exist, matching load_runtime_records and the data loader.
        for candidate in ("_id", "id"):
            if candidate in columns:
                return candidate
        raise ValueError(
            "HotpotQA dataset is missing an ID column ('_id' or 'id'); "
            f"available columns: {sorted(columns)}"
        )
    raise ValueError(f"unsupported annotation dataset {dataset!r}")


def load_runtime_records(config: Mapping[str, Any], split_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read only IDs, questions, and context documents; never request label columns."""
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        wanted = [str(value) for value in split_manifest["datasets"][dataset]["evaluation_ids"]]
        wanted_set = set(wanted)
        raw = _raw_dataset(config, dataset)
        id_column = _dataset_id_column(dataset, raw)
        keep = ("id", "question", "title", "context") if dataset == "squad" else (id_column, "question", "context")
        rows = raw.select_columns(list(keep))
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            query_id = str(row[id_column])
            if query_id not in wanted_set:
                continue
            if dataset == "squad":
                title, text = str(row.get("title") or ""), str(row["context"])
                docs = [{"doc_id": _document_id("squad", title, text), "title": title, "text": text}]
            else:
                docs = []
                for title, sentences in _hotpot_context_entries(row.get("context"), len(by_id)):
                    text = " ".join(str(sentence) for sentence in sentences)
                    docs.append({"doc_id": _document_id("hotpot", title, text), "title": title, "text": text})
            by_id[query_id] = {
                "id": query_id, "dataset": dataset, "question": str(row["question"]), "documents": docs,
            }
        if len(by_id) != len(wanted):
            missing = len(wanted_set - set(by_id))
            raise ValueError(f"{dataset} runtime data missing {missing} selected query records")
        output.extend(by_id[query_id] for query_id in wanted)
    pairs = [(record["dataset"], record["id"]) for record in output]
    expected_pairs = [tuple(pair) for pair in split_manifest["ordered_dataset_query_ids"]]
    if pairs != expected_pairs:
        raise ValueError("runtime records do not match the exact split manifest order")
    if any(set(record) != {"id", "dataset", "question", "documents"} for record in output):
        raise RuntimeError("runtime record contains unexpected label-bearing fields")
    return output


def _iter_evaluation_annotation_batches(
    config: Mapping[str, Any], split_manifest: Mapping[str, Any], batch_size: int = 250,
):
    """Yield exact annotation batches; only call after all context selection ends."""
    if batch_size <= 0:
        raise ValueError("annotation batch size must be positive")
    for dataset in DATASETS:
        wanted_ids = [str(value) for value in split_manifest["datasets"][dataset]["evaluation_ids"]]
        ids = set(wanted_ids)
        if len(ids) != len(wanted_ids):
            raise ValueError(f"{dataset} split contains duplicate selected query IDs")
        raw = _raw_dataset(config, dataset)
        id_column = _dataset_id_column(dataset, raw)
        selected = raw.filter(lambda row: str(row[id_column]) in ids)
        source_ids = [str(row[id_column]) for row in selected]
        source_seen: set[str] = set()
        source_duplicates = []
        for query_id in source_ids:
            if query_id in source_seen and query_id not in source_duplicates:
                source_duplicates.append(query_id)
            source_seen.add(query_id)
        if source_duplicates:
            raise ValueError(
                f"{dataset} source contains duplicate annotation IDs: {source_duplicates[:5]}"
            )
        row_index_by_id = {str(row[id_column]): index for index, row in enumerate(selected)}
        missing_source = sorted(ids - set(row_index_by_id))
        unexpected_source = sorted(set(row_index_by_id) - ids)
        if missing_source or unexpected_source:
            raise ValueError(
                f"{dataset} source annotation IDs do not exactly match the selected split: "
                f"missing={missing_source[:5]}, unexpected={unexpected_source[:5]}"
            )
        for offset in range(0, len(wanted_ids), batch_size):
            batch_ids = wanted_ids[offset:offset + batch_size]
            batch_rows = selected.select([row_index_by_id[query_id] for query_id in batch_ids])
            normalized = _squad_records(batch_rows) if dataset == "squad" else _hotpot_records(batch_rows)
            normalized_ids = [str(record["id"]) for record in normalized]
            if len(normalized_ids) != len(set(normalized_ids)):
                duplicates = sorted({query_id for query_id in normalized_ids if normalized_ids.count(query_id) > 1})
                raise ValueError(f"{dataset} has duplicate normalized annotation IDs: {duplicates[:5]}")
            by_id = {str(record["id"]): record for record in normalized}
            missing = sorted(set(batch_ids) - set(by_id))
            extra = sorted(set(by_id) - set(batch_ids))
            if missing or extra:
                raise ValueError(
                    f"{dataset} annotation batch IDs do not exactly match the selected split: "
                    f"missing={missing[:5]}, unexpected={extra[:5]}"
                )
            yield dataset, [by_id[query_id] for query_id in batch_ids]


def _load_evaluation_annotations(config: Mapping[str, Any], split_manifest: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Compatibility loader for tests/tools; the large run uses batches."""
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, batch in _iter_evaluation_annotation_batches(config, split_manifest):
        for record in batch:
            key = (dataset, str(record["id"]))
            if key in result:
                raise ValueError(f"duplicate evaluation annotation record: {key}")
            result[key] = record
    expected_keys = {
        (dataset, str(query_id))
        for dataset in DATASETS
        for query_id in split_manifest["datasets"][dataset]["evaluation_ids"]
    }
    if set(result) != expected_keys:
        missing = sorted(expected_keys - set(result))
        extra = sorted(set(result) - expected_keys)
        raise ValueError(f"evaluation annotation mapping is not exact: missing={missing[:5]}, unexpected={extra[:5]}")
    return result


def _chunk_hash(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(json.dumps(chunk, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class _PrecomputedSentenceEncoder:
    """Serve normalized sentence vectors to the unchanged semantic chunker."""

    def __init__(self, dimension: int, vectors: Mapping[str, np.ndarray]):
        self.dimension = int(dimension)
        self.vectors = vectors

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences, **kwargs):
        try:
            return np.asarray([self.vectors[str(sentence)] for sentence in sentences], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(f"sentence was not included in the batch preflight: {exc.args[0][:80]}") from exc


def _prime_sentence_vectors(
    corpus: list[dict[str, Any]], encoder: SentenceTransformer, cache_dir: Path, model_name: str,
) -> tuple[_PrecomputedSentenceEncoder, dict[str, Any]]:
    """Batch missing sentence vectors, then delegate boundaries to semantic_chunks."""
    dimension = int(encoder.get_sentence_embedding_dimension())
    vectors_by_text: dict[str, np.ndarray] = {}
    missing: list[str] = []
    missing_seen: set[str] = set()
    sentence_count = 0
    cache_hits = 0
    for document in corpus:
        sentences = [part.text for part in segment_sentences(str(document["text"]))]
        sentence_count += len(sentences)
        cache_key = str(model_name) + "\0" + str(dimension) + "\0" + "\0".join(sentences)
        path = cache_dir / f"sentences-{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()}.npy"
        if path.is_file():
            cached = np.load(path, allow_pickle=False)
            if cached.shape != (len(sentences), dimension):
                raise ValueError(f"sentence cache has unexpected shape: {path.name}")
            cache_hits += 1
            for sentence, vector in zip(sentences, cached):
                vectors_by_text.setdefault(sentence, np.asarray(vector, dtype=np.float32))
        else:
            for sentence in sentences:
                if sentence not in vectors_by_text and sentence not in missing_seen:
                    missing_seen.add(sentence)
                    missing.append(sentence)
    if missing:
        batch_size = 64
        batch_vectors = np.asarray(encoder.encode(
            missing, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=True,
        ), dtype=np.float32)
        if batch_vectors.shape != (len(missing), dimension) or not np.isfinite(batch_vectors).all():
            raise ValueError("batched sentence embedding output is malformed")
        if not np.allclose(np.linalg.norm(batch_vectors, axis=1), 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("batched sentence embeddings are not normalized")
        for sentence, vector in zip(missing, batch_vectors):
            vectors_by_text[sentence] = vector
    verify_texts = missing[:min(16, len(missing))]
    max_abs_difference = 0.0
    if verify_texts:
        direct = np.asarray(encoder.encode(
            verify_texts, batch_size=1, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ), dtype=np.float32)
        batched = np.asarray([vectors_by_text[text] for text in verify_texts], dtype=np.float32)
        max_abs_difference = float(np.max(np.abs(direct - batched)))
        if not np.allclose(direct, batched, rtol=1e-4, atol=1e-5):
            raise ValueError("batched sentence vectors differ from single-input vectors beyond tolerance")
    return _PrecomputedSentenceEncoder(dimension, vectors_by_text), {
        "sentence_count_with_document_repeats": sentence_count,
        "unique_sentence_count": len(vectors_by_text),
        "new_unique_sentences_encoded": len(missing),
        "existing_document_cache_hits": cache_hits,
        "batch_size": 64,
        "verified_single_vs_batch_max_abs_difference": max_abs_difference,
        "single_vs_batch_tolerance": {"rtol": 1e-4, "atol": 1e-5},
    }


def _write_preflight_inventory(path: Path, chunks: list[dict[str, Any]]) -> None:
    _write_jsonl(path, chunks)


def preflight(config_path: str | Path) -> dict[str, Any]:
    """Prepare full-corpus chunk inventories and validate limits, without retrieval/evaluation."""
    root, config_file, config = _root_and_config(config_path)
    results_dir, processed_dir, cache_dir = validate_output_locations(root, config, preflight=True)
    split_manifest, split_hash = validate_split_file(root, config)
    runtime_records = load_runtime_records(config, split_manifest)
    corpus = _unique_corpus(runtime_records)
    if not corpus:
        raise ValueError("expanded evaluation corpus is empty")

    tokenizer_name = str(config["tokenizer_name"])
    model_name = str(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("fast tokenizer required")
    encoder = SentenceTransformer(model_name, device=str(config["device"]))
    max_length = int(config["embedding_max_sequence_length"])
    encoder.max_seq_length = max_length
    m2_size, special_count = encoder_content_token_limit(tokenizer, max_length)
    if m2_size != 510:
        raise ValueError(f"M2 must remain at 510 content tokens; computed {m2_size}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    sentence_encoder, sentence_batch_audit = _prime_sentence_vectors(corpus, encoder, cache_dir, model_name)

    m1_chunks: list[dict[str, Any]] = []
    m2_chunks: list[dict[str, Any]] = []
    m3_chunks: list[dict[str, Any]] = []
    started = perf_counter()
    for document in corpus:
        m1_chunks.extend(fixed_size_chunks(document["text"], document["doc_id"], tokenizer, 256, 32))
        m2_chunks.extend(fixed_size_chunks(document["text"], document["doc_id"], tokenizer, m2_size, 32))
        m3_chunks.extend(semantic_chunks(
            document["text"], document["doc_id"], tokenizer, sentence_encoder,
            float(config["semantic_similarity_threshold"]), int(config["semantic_min_tokens"]),
            int(config["semantic_max_tokens"]), cache_dir, model_name,
        ))
    chunking_ms = (perf_counter() - started) * 1000.0
    chunks_by_method = {M1: m1_chunks, M2: m2_chunks, M3: m3_chunks}
    audits = {
        method: audit_chunk_embedding_lengths(chunks, tokenizer, max_length)
        for method, chunks in chunks_by_method.items()
    }
    if any(audit["over_limit_chunk_count"] for audit in audits.values()):
        raise ValueError("preflight found chunks over the model sequence limit")
    if encoder.get_sentence_embedding_dimension() != 384:
        raise ValueError("embedding model dimension differs from the frozen MiniLM protocol")

    processed_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = processed_dir / "chunks"
    chunks_dir.mkdir()
    embeddings_dir = processed_dir / "embeddings"
    indexes_dir = processed_dir / "indexes"
    embeddings_dir.mkdir()
    indexes_dir.mkdir()
    paths = {}
    for method, chunks in chunks_by_method.items():
        path = chunks_dir / f"{method}.jsonl"
        _write_preflight_inventory(path, chunks)
        paths[method] = {
            "relative_path": path.relative_to(processed_dir).as_posix(),
            "chunk_count": len(chunks), "chunk_sha256": _chunk_hash(chunks),
            "file_bytes": path.stat().st_size,
        }
    documents_by_dataset = {
        dataset: sum(str(document["doc_id"]).startswith(f"{prefix}:") for document in corpus)
        for dataset, prefix in (("squad", "squad"), ("hotpotqa", "hotpot"))
    }
    embedding_dimension = int(encoder.get_sentence_embedding_dimension())
    total_vectors = sum(len(chunks) for chunks in chunks_by_method.values())
    vector_bytes = total_vectors * embedding_dimension * 4
    manifest = {
        "run_id": config["run_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_path": _portable_repo_path(config_file, root),
        "configuration_sha256": _sha256_file(config_file),
        "split_manifest_path": _portable_repo_path(root / config["split_manifest"], root),
        "split_manifest_sha256": _sha256_file(root / config["split_manifest"]),
        "ordered_dataset_query_ids_sha256": split_hash,
        "evaluation_query_count": 10_000, "dataset_counts": {"squad": 5_000, "hotpotqa": 5_000},
        "unique_corpus_document_count": len(corpus), "unique_corpus_document_counts": documents_by_dataset,
        "chunk_counts": {method: len(chunks) for method, chunks in chunks_by_method.items()},
        "chunk_artifacts": paths, "embedding_model": model_name,
        "embedding_dimension": embedding_dimension, "embedding_vector_count_across_indexes": total_vectors,
        "float32_embedding_payload_bytes_across_indexes": vector_bytes,
        "estimated_faiss_index_payload_bytes": vector_bytes,
        "estimated_in_memory_vectors_and_faiss_bytes": vector_bytes * 2,
        "m2_content_tokens": m2_size, "tokenizer_special_token_count": special_count,
        "embedding_input_audit": audits, "full_corpus_chunking_latency_ms": chunking_ms,
        "sentence_embedding_batch_audit": sentence_batch_audit,
        "preflight_only": True, "retrievals_run": 0, "evaluation_metrics_computed": False,
        "runtime_data_fields": ["id", "dataset", "question", "documents"],
        "answer_or_evidence_columns_loaded": False,
        "output_locations": {
            "processed": _portable_repo_path(processed_dir, root),
            "cache": _portable_repo_path(cache_dir, root),
            "results_reserved": _portable_repo_path(results_dir, root),
        },
    }
    (processed_dir / "preflight_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def _validate_preflight_inventory_files(
    root: Path, config: Mapping[str, Any], split_manifest: Mapping[str, Any],
    processed_dir: Path, document_text: Mapping[str, str],
) -> dict[str, Any]:
    """Validate inventories one line at a time without retaining three copies."""
    manifest = json.loads((processed_dir / "preflight_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("ordered_dataset_query_ids_sha256") != split_manifest["ordered_dataset_query_ids_sha256"]:
        raise ValueError("preflight split hash differs from current split manifest")
    split_manifest_path = root / config["split_manifest"]
    if manifest.get("split_manifest_sha256") != _sha256_file(split_manifest_path):
        raise ValueError("preflight split manifest file hash mismatch")
    if manifest.get("configuration_sha256") != _sha256_file(root / "configs/day3_heldout_10000_m4v3.yaml"):
        raise ValueError("preflight configuration hash mismatch")
    expected_docs = set(document_text)
    for method in (M1, M2, M3):
        metadata = manifest["chunk_artifacts"][method]
        path = processed_dir / metadata["relative_path"]
        digest = hashlib.sha256()
        actual_docs: set[str] = set()
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                digest.update(json.dumps(
                    chunk, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8"))
                digest.update(b"\n")
                doc_id = str(chunk.get("doc_id", ""))
                source = document_text.get(doc_id)
                if source is None:
                    raise ValueError(f"{method} contains an unknown document ID: {doc_id}")
                start, end = int(chunk["start"]), int(chunk["end"])
                if start < 0 or end < start or source[start:end] != str(chunk["text"]):
                    raise ValueError(f"{method} chunk source offset validation failed: {chunk.get('chunk_id')}")
                actual_docs.add(doc_id)
                count += 1
        if digest.hexdigest() != metadata["chunk_sha256"]:
            raise ValueError(f"{method} chunk inventory hash mismatch")
        if count != int(metadata["chunk_count"]):
            raise ValueError(f"{method} chunk inventory count mismatch")
        if actual_docs != expected_docs:
            raise ValueError(f"{method} chunk corpus differs from selected document corpus")
    return manifest


def _search_response_payload(record: Mapping[str, Any], response: SearchResponse) -> dict[str, Any]:
    return {
        "dataset": str(record["dataset"]), "query_id": str(record["id"]),
        "query_sha256": hashlib.sha256(str(record["question"]).encode("utf-8")).hexdigest(),
        "query_embedding": np.asarray(response.query_embedding, dtype=np.float32).tolist(),
        "latency_ms": float(response.latency_ms),
        "hits": [
            {"chunk": hit.chunk, "score": float(hit.score), "rank": int(hit.rank)}
            for hit in response.hits
        ],
    }


def _response_from_payload(payload: Mapping[str, Any]) -> SearchResponse:
    return SearchResponse(
        [SearchHit(dict(hit["chunk"]), float(hit["score"]), int(hit["rank"])) for hit in payload["hits"]],
        np.asarray(payload["query_embedding"], dtype=np.float32),
        float(payload["latency_ms"]),
    )


def _persist_index_artifacts(retriever: DenseRetriever, method: str, processed_dir: Path) -> None:
    indexes_dir = processed_dir / "indexes"
    embeddings_dir = processed_dir / "embeddings"
    retriever.save(indexes_dir / f"{method}.faiss")
    np.save(embeddings_dir / f"{method}.npy", retriever.vectors, allow_pickle=False)
    _write_jsonl(indexes_dir / f"{method}_rows.jsonl", (
        {"index_row": index, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]}
        for index, chunk in enumerate(retriever.chunks)
    ))


def _build_method_search_spool(
    method: str, chunks: list[dict[str, Any]], query_records: list[dict[str, Any]],
    encoder: SentenceTransformer, model_name: str, top_k: int, processed_dir: Path,
    spool_path: Path,
) -> float:
    """Build one dense index, persist exact top-k responses, then release it."""
    if spool_path.exists():
        raise FileExistsError(f"refusing to overwrite retrieval spool: {spool_path}")
    retriever = DenseRetriever(encoder, model_name)
    started = perf_counter()
    try:
        retriever.build(chunks)
        build_latency_ms = (perf_counter() - started) * 1000.0
        _persist_index_artifacts(retriever, method, processed_dir)
        with spool_path.open("x", encoding="utf-8") as handle:
            for record in query_records:
                response = retriever.retrieve(str(record["question"]), top_k=top_k)
                handle.write(json.dumps(
                    _search_response_payload(record, response), ensure_ascii=False,
                ) + "\n")
        return build_latency_ms
    finally:
        # The caller retains no retriever reference; dropping this local releases
        # its vectors, FAISS index, chunk copies, and position map before next method.
        del retriever
        gc.collect()


class _SearchResponseReader:
    """Stream stored M1/M2 search responses in the frozen query order."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("r", encoding="utf-8")
        self.count = 0

    def next_for(self, record: Mapping[str, Any]) -> SearchResponse:
        line = self.handle.readline()
        if not line:
            raise ValueError(f"retrieval spool ended early: {self.path.name}")
        payload = json.loads(line)
        expected_hash = hashlib.sha256(str(record["question"]).encode("utf-8")).hexdigest()
        if (payload.get("dataset") != record["dataset"] or payload.get("query_id") != record["id"] or
                payload.get("query_sha256") != expected_hash):
            raise ValueError(f"retrieval spool query order/hash mismatch in {self.path.name} at row {self.count}")
        self.count += 1
        return _response_from_payload(payload)

    def assert_exhausted(self) -> None:
        if self.handle.readline():
            raise ValueError(f"retrieval spool contains unexpected extra rows: {self.path.name}")

    def close(self) -> None:
        self.handle.close()


class _SingleSearchResponseRetriever:
    """Adapter for the validated six-method selector's existing retrieve call."""

    def __init__(self, response: SearchResponse, query: str, top_k: int):
        self.response = response
        self.query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        self.top_k = int(top_k)
        self.used = False

    def retrieve(self, query: str, top_k: int = 5) -> SearchResponse:
        if self.used:
            raise RuntimeError("precomputed retrieval response was requested more than once")
        if hashlib.sha256(query.encode("utf-8")).hexdigest() != self.query_sha256 or int(top_k) != self.top_k:
            raise ValueError("precomputed retrieval response does not match the current query/top-k")
        self.used = True
        return self.response


def _selection_item_for_spool(item: Mapping[str, Any]) -> dict[str, Any]:
    selections = {}
    for method, selection in item["selections"].items():
        decisions = selection["candidate_decisions"]
        rejected_count = sum(not bool(decision.get("accepted")) for decision in decisions)
        selections[method] = {
            "ranked_chunks": [
                {key: chunk[key] for key in ("chunk_id", "doc_id", "start", "end")}
                for chunk in selection["ranked_chunks"]
            ],
            "selected_chunks": selection["selected_chunks"],
            "seed_chunk_ids": selection["seed_chunk_ids"],
            "seed_ranks": selection["seed_ranks"],
            "seed_scores": selection["seed_scores"],
            "query_embedding_sha256": selection["query_embedding_sha256"],
            "context_text": selection["context_text"],
            "context_token_count": selection["context_token_count"],
            "token_budget": selection["token_budget"],
            "expanded_chunk_ids": selection["expanded_chunk_ids"],
            "similarity_scores": selection["similarity_scores"],
            "expansion_latency_ms": selection["expansion_latency_ms"],
            "candidate_decision_count": len(decisions),
            "candidate_rejected_count": rejected_count,
            "budget_excluded_chunk_ids": selection["budget_excluded_chunk_ids"],
            "retrieval_latency_ms": float(selection["search"].latency_ms),
        }
    return {
        "query_id": item["query_id"], "dataset": item["dataset"],
        "query_text": item["query_text"], "classification": item["classification"],
        "selections": selections,
    }


def _selection_item_from_spool(item: Mapping[str, Any]) -> dict[str, Any]:
    restored = {}
    for method, source in item["selections"].items():
        decisions = ([{"accepted": False}] * int(source["candidate_rejected_count"]) +
                     [{"accepted": True}] * (int(source["candidate_decision_count"]) -
                                               int(source["candidate_rejected_count"])))
        restored[method] = {
            **source,
            "search": SearchResponse([], np.empty((0,), dtype=np.float32),
                                     float(source["retrieval_latency_ms"])),
            "candidate_decisions": decisions,
        }
        restored[method].pop("retrieval_latency_ms", None)
        restored[method].pop("candidate_decision_count", None)
        restored[method].pop("candidate_rejected_count", None)
    return {
        "query_id": item["query_id"], "dataset": item["dataset"],
        "query_text": item["query_text"], "classification": item["classification"],
        "selections": restored,
    }


def _sha256_path(path: Path) -> str:
    return _sha256_file(path)


class _StreamingResultWriter:
    """Write complete per-query outputs without retaining their text in memory."""

    def __init__(self, jsonl_path: Path, csv_path: Path):
        self.json_handle = jsonl_path.open("x", encoding="utf-8")
        self.csv_handle = csv_path.open("x", encoding="utf-8-sig", newline="")
        self.csv_writer = None

    def write(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.json_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            csv_row = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            if self.csv_writer is None:
                self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=list(csv_row))
                self.csv_writer.writeheader()
            self.csv_writer.writerow(csv_row)

    def close(self) -> None:
        self.json_handle.close()
        self.csv_handle.close()


def _load_gold_annotations_after_selection(config: Mapping[str, Any], split_manifest: Mapping[str, Any]):
    return _load_evaluation_annotations(config, split_manifest)


def run_evaluation(config_path: str | Path, executed_command: str | None = None) -> Path:
    """Run the full 10,000-query evaluation. CLI requires explicit opt-in."""
    root, config_file, config = _root_and_config(config_path)
    results_dir, processed_dir, cache_dir = validate_output_locations(root, config, preflight=False)
    split_manifest, split_hash = validate_split_file(root, config)
    runtime_records = load_runtime_records(config, split_manifest)
    corpus = _unique_corpus(runtime_records)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], use_fast=True)
    encoder = SentenceTransformer(config["model_name"], device=str(config["device"]))
    max_length = int(config["embedding_max_sequence_length"])
    encoder.max_seq_length = max_length
    document_text = {str(row["doc_id"]): str(row["text"]) for row in corpus}
    preflight_manifest = _validate_preflight_inventory_files(
        root, config, split_manifest, processed_dir, document_text,
    )
    query_records = [
        {"id": record["id"], "dataset": record["dataset"], "question": record["question"]}
        for record in runtime_records
    ]
    if len(query_records) != 10_000:
        raise RuntimeError("runtime loader did not produce exactly 10,000 query-only records")
    # Corpus texts are needed only for artifact offset validation; query selection
    # consumes only IDs, dataset names, and question text from this point forward.
    del runtime_records, corpus, document_text
    gc.collect()

    indexes_dir = processed_dir / "indexes"
    embeddings_dir = processed_dir / "embeddings"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    if any(indexes_dir.iterdir()) or any(embeddings_dir.iterdir()):
        raise FileExistsError("refusing to overwrite an existing 10,000-query index or embedding artifact")
    retrieval_dir = processed_dir / "retrievals"
    selection_spool_path = processed_dir / "selection_spool.jsonl"
    staging_dir = processed_dir / "evaluation_staging"
    if retrieval_dir.exists() or selection_spool_path.exists() or staging_dir.exists():
        raise FileExistsError("refusing to overwrite existing sequential-evaluation spool/staging artifacts")
    retrieval_dir.mkdir()
    staging_dir.mkdir()

    index_build_ms = {}
    rules = {name: base.ExpansionRule(int(rule["max_depth"]), int(rule["max_expansions"]))
             for name, rule in config["expansion_rules"].items()}
    token_budget = int(config["token_budget"])
    top_k = int(config["top_k"])
    v3_config = base.ExpansionV3Config(**config["m4_v3"])
    retrieval_spool_hashes: dict[str, str] = {}

    # M1 and M2 are built, searched for every query, and released in turn. Their
    # exact SearchResponse values (including float32 query vectors) are persisted.
    for method in (M1, M2):
        chunks = _jsonl_read(processed_dir / "chunks" / f"{method}.jsonl")
        audit = audit_chunk_embedding_lengths(chunks, tokenizer, max_length)
        if audit["over_limit_chunk_count"]:
            raise ValueError(f"{method} preflight chunks exceed encoder length")
        spool_path = retrieval_dir / f"{method}.jsonl"
        index_build_ms[method] = _build_method_search_spool(
            method, chunks, query_records, encoder, str(config["model_name"]), top_k,
            processed_dir, spool_path,
        )
        retrieval_spool_hashes[method] = _sha256_path(spool_path)
        del chunks
        gc.collect()

    # M3 stays resident only while all M3 retrieval and expansion decisions are
    # made. Its vector map is constructed once and shared across all queries.
    m3_chunks = _jsonl_read(processed_dir / "chunks" / f"{M3}.jsonl")
    m3_audit = audit_chunk_embedding_lengths(m3_chunks, tokenizer, max_length)
    if m3_audit["over_limit_chunk_count"]:
        raise ValueError("M3 preflight chunks exceed encoder length")
    m3_retriever = DenseRetriever(encoder, str(config["model_name"]))
    started = perf_counter()
    m3_retriever.build(m3_chunks)
    index_build_ms[M3] = (perf_counter() - started) * 1000.0
    _persist_index_artifacts(m3_retriever, M3, processed_dir)
    m3_chunks = m3_retriever.chunks
    chunk_embeddings = base.chunk_embedding_map_from_retriever(m3_retriever)
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in m3_chunks}

    m1_reader = _SearchResponseReader(retrieval_dir / f"{M1}.jsonl")
    m2_reader = _SearchResponseReader(retrieval_dir / f"{M2}.jsonl")
    class_counts: dict[str, int] = {}
    selected_query_count = 0
    candidate_log_path = staging_dir / "candidate_decisions.jsonl"
    with selection_spool_path.open("x", encoding="utf-8") as selection_handle, \
            candidate_log_path.open("x", encoding="utf-8") as decisions_handle:
        for record in query_records:
            query = str(record["question"])
            classification = base.classify_query(query)
            complexity = classification["complexity"]
            class_counts[complexity] = class_counts.get(complexity, 0) + 1
            m1_response = m1_reader.next_for(record)
            m2_response = m2_reader.next_for(record)
            selections = base.select_contexts_for_query(
                query, complexity, {M3: m3_chunks},
                {
                    M1: _SingleSearchResponseRetriever(m1_response, query, top_k),
                    M2: _SingleSearchResponseRetriever(m2_response, query, top_k),
                    M3: m3_retriever,
                },
                tokenizer, rules, token_budget,
                float(config["minimum_neighbor_similarity"]),
                float(config["m4_v2"]["redundancy_threshold"]), v3_config, top_k,
                chunk_embeddings=chunk_embeddings, chunk_by_id=chunk_by_id,
            )
            item = {
                "query_id": str(record["id"]), "dataset": str(record["dataset"]),
                "query_text": query, "classification": classification, "selections": selections,
            }
            if any(int(selection["context_token_count"]) > token_budget for selection in selections.values()):
                raise RuntimeError(f"query {record['id']} produced a context over the hard token budget")
            selection_handle.write(json.dumps(_selection_item_for_spool(item), ensure_ascii=False) + "\n")
            for method in METHODS:
                selection = selections[method]
                decisions_handle.write(json.dumps({
                    "query_id": item["query_id"], "dataset": item["dataset"], "method": method,
                    "seed_chunk_ids": selection["seed_chunk_ids"], "seed_ranks": selection["seed_ranks"],
                    "seed_scores": selection["seed_scores"],
                    "query_embedding_sha256": selection["query_embedding_sha256"],
                    "decisions": selection["candidate_decisions"],
                    "expanded_chunk_ids": selection["expanded_chunk_ids"],
                    "context_token_count": selection["context_token_count"],
                    "token_budget": selection["token_budget"],
                }, ensure_ascii=False) + "\n")
            selected_query_count += 1
            del selections, item, m1_response, m2_response
    m1_reader.assert_exhausted()
    m2_reader.assert_exhausted()
    m1_reader.close()
    m2_reader.close()
    if selected_query_count != 10_000:
        raise RuntimeError(f"only {selected_query_count} of 10,000 contexts were selected")
    selection_spool_hash = _sha256_path(selection_spool_path)

    # All expansion decisions are complete. The M3 vector map/index/model and
    # query list are now unnecessary; release them before evaluation annotations.
    embedding_dimension = int(encoder.get_sentence_embedding_dimension())
    del chunk_embeddings, chunk_by_id, m3_retriever, m3_chunks, encoder, tokenizer, query_records
    gc.collect()

    metric_rows: list[dict[str, Any]] = []
    result_row_count = 0
    completed_query_count = 0
    all_contexts_within_budget = True
    maximum_context_tokens = 0
    selection_handle = selection_spool_path.open("r", encoding="utf-8")
    result_writer = _StreamingResultWriter(
        staging_dir / "per_query.jsonl", staging_dir / "per_query.csv",
    )
    try:
        # Evaluation labels are loaded only now, in bounded batches. Results are
        # serialized immediately; only small numeric summaries remain in RAM.
        for dataset, annotation_batch in _iter_evaluation_annotation_batches(
            config, split_manifest, batch_size=250,
        ):
            for annotation in annotation_batch:
                line = selection_handle.readline()
                if not line:
                    raise ValueError("selection spool ended before all annotation batches were scored")
                stored = json.loads(line)
                if stored["dataset"] != dataset or str(stored["query_id"]) != str(annotation["id"]):
                    raise ValueError("selection spool and annotation batch query IDs are out of order")
                pending_item = _selection_item_from_spool(stored)
                rows = base.score_selected_contexts([pending_item], [annotation], token_budget)
                if len(rows) != len(METHODS):
                    raise RuntimeError(f"query {annotation['id']} did not produce six method rows")
                result_writer.write(rows)
                for row in rows:
                    result_row_count += 1
                    all_contexts_within_budget = all_contexts_within_budget and (
                        int(row["context_token_count"]) <= int(row["token_budget"])
                    )
                    maximum_context_tokens = max(maximum_context_tokens, int(row["context_token_count"]))
                    metric_rows.append({
                        key: row[key] for key in (
                            "query_id", "dataset", "method", "recall_at_1", "recall_at_3",
                            "recall_at_5", "mrr", "context_precision", "context_token_count",
                            "retrieval_latency_ms", "expansion_latency_ms",
                        )
                    })
                completed_query_count += 1
                del rows, pending_item, stored, annotation
            del annotation_batch
        if selection_handle.readline():
            raise ValueError("selection spool contains queries without matching evaluation annotations")
    finally:
        selection_handle.close()
        result_writer.close()
    if completed_query_count != 10_000 or result_row_count != 60_000:
        raise RuntimeError(
            f"incomplete output: {completed_query_count} queries and {result_row_count} method rows"
        )
    if not all_contexts_within_budget:
        raise RuntimeError("one or more selected contexts exceeded the configured token budget")

    aggregate = _aggregate(metric_rows)
    _write_csv(staging_dir / "aggregate_metrics.csv", aggregate)
    paired = []
    for method in METHODS:
        if method != M3:
            paired.extend(base.paired_bootstrap_comparisons(
                metric_rows, method_a=M3, method_b=method,
                repeats=int(config["bootstrap_repeats"]), seed=int(config["seed"]),
            ))
    _write_csv(staging_dir / "paired_bootstrap_ci.csv", paired)
    plots = base._write_six_method_plots(aggregate, staging_dir / "plots")
    manifest = {
        "run_id": config["run_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed_command": executed_command, "methods": list(METHODS),
        "evaluation_query_count": 10_000, "completed_query_count": completed_query_count, "skipped_query_count": 0,
        "dataset_counts": {"squad": 5_000, "hotpotqa": 5_000},
        "split_manifest_path": str(root / config["split_manifest"]),
        "split_manifest_sha256": _sha256_file(root / config["split_manifest"]),
        "ordered_dataset_query_ids_sha256": split_hash,
        "heldout_query_ids_sha256": split_hash,
        "unique_corpus_document_count": preflight_manifest["unique_corpus_document_count"],
        "chunk_counts": dict(preflight_manifest["chunk_counts"]),
        "semantic_chunk_count": preflight_manifest["chunk_counts"][M3],
        "embedding_dimension": embedding_dimension,
        "index_build_latency_ms": index_build_ms,
        "memory_execution_strategy": {
            "maximum_simultaneous_faiss_indexes": 1,
            "m1_m2_retrieval_responses_spooled": True,
            "m1_m2_retrievers_released_after_search": True,
            "m3_vector_map_built_once": True,
            "m3_index_released_after_all_expansion": True,
            "selection_results_spooled_per_query": True,
            "per_query_outputs_written_incrementally": True,
            "annotation_batch_size": 250,
        },
        "retrieval_spool_sha256": retrieval_spool_hashes,
        "selection_spool_sha256": selection_spool_hash,
        "result_row_count": result_row_count,
        "m3_seed_ids_ranks_scores_and_query_embedding_shared": True,
        "gold_annotations_loaded_after_context_selection": True,
        "gold_evidence_runtime_inputs": False,
        "all_contexts_within_budget": all_contexts_within_budget,
        "maximum_context_tokens": maximum_context_tokens,
        "classifier_prediction_counts": class_counts,
        "configuration_sha256": _sha256_file(config_file),
        "output_locations": {"results": str(results_dir), "processed": str(processed_dir), "cache": str(cache_dir)},
        "outputs": ["per_query.csv", "per_query.jsonl", "aggregate_metrics.csv", "paired_bootstrap_ci.csv",
                    "candidate_decisions.jsonl", "experiment_manifest.json", *[str(p.relative_to(staging_dir)) for p in plots]],
    }
    base_hashes = base.integrity_hashes(
        root, config_file, source_paths=SOURCE_HASH_PATHS,
    )
    manifest.update(base_hashes)
    report = base._report_text(metric_rows, manifest)
    # The shared report formatter was originally written for the 400-query run.
    report = report.replace(
        f"- Held-out queries: {manifest['evaluation_query_count']} (200 SQuAD, 200 HotpotQA)",
        "- Held-out queries: 10,000 (5,000 SQuAD, 5,000 HotpotQA)",
    )
    (staging_dir / "heldout_evaluation_report.md").write_text(report, encoding="utf-8")
    (staging_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if results_dir.exists():
        raise FileExistsError(f"refusing to overwrite results directory: {results_dir}")
    os.replace(staging_dir, results_dir)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run the isolated 10,000-query six-method evaluation")
    parser.add_argument("--config", default="configs/day3_heldout_10000_m4v3.yaml")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create-split", action="store_true")
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--run-evaluation", action="store_true")
    args = parser.parse_args()
    if args.create_split:
        path, manifest = create_split(args.config)
        print(f"Created 10,000-query ID manifest: {path}; hash={manifest['ordered_dataset_query_ids_sha256']}")
    elif args.preflight_only:
        manifest = preflight(args.config)
        print(json.dumps({key: manifest[key] for key in (
            "evaluation_query_count", "dataset_counts", "unique_corpus_document_count",
            "unique_corpus_document_counts", "chunk_counts", "float32_embedding_payload_bytes_across_indexes",
            "full_corpus_chunking_latency_ms", "embedding_input_audit",
        )}, indent=2))
    else:
        command = subprocess.list2cmdline([
            sys.executable, "-m", "query_adaptive_rag.evaluation.day3_heldout_10000_m4v3", *sys.argv[1:],
        ])
        print(f"Full 10,000-query evaluation completed: {run_evaluation(args.config, command)}")


if __name__ == "__main__":
    main()
