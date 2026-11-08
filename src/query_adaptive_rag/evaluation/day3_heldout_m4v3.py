"""Isolated six-method held-out evaluation including corrected M4-v3.

This module deliberately leaves the validated four-method Day 3 runner and all
method implementations untouched. M4, M4-v2, and M4-v3 receive one shared M3
SearchResponse per query. Gold evidence is read only after every context for
the held-out set has been selected.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import yaml

from ..chunking.fixed_size import fixed_size_chunks
from ..chunking.text_utils import OffsetTokenizer, count_tokens
from ..classification.complexity import classify_query
from ..config import set_random_seed
from ..evaluation.context import context_source_chunks
from ..evaluation.evidence import deduplicate_evidence
from ..evaluation.metrics import calculate_metrics
from ..evaluation.protocol import method_protocol
from ..evaluation.splits import deterministic_split
from ..evaluation.statistics import paired_bootstrap_comparisons
from ..evaluation.token_limits import audit_chunk_embedding_lengths, encoder_content_token_limit
from ..expansion.context_expander import ExpansionRule, TokenBudgetError, assemble_context, expand_context
from ..expansion.context_expander_v2 import chunk_embedding_map_from_retriever, expand_context_v2
from ..expansion.context_expander_v3 import ExpansionV3Config, expand_context_v3
from ..retrieval.dense import DenseRetriever, SearchResponse
from .day3 import (
    DATASET_COMPLEXITY,
    _aggregate,
    _budget_context,
    _dependency_versions,
    _heldout_records,
    _unique_corpus,
    _write_csv,
    _write_jsonl,
)


M1 = "M1_fixed_256"
M2 = "M2_fixed_510"
M3 = "M3_semantic"
M4 = "M4_adaptive"
M4V2 = "M4_v2_redundancy"
M4V3 = "M4_v3_fact_alias_fix"
METHODS = (M1, M2, M3, M4, M4V2, M4V3)
EXPANSION_METHODS = (M4, M4V2, M4V3)
EXPECTED_SPLIT_HASH = "762bd07ae47cc0a750046e04c1eb64697875b0643e431cc0af3cee6b5adf8bab"
HASHED_SOURCE_PATHS = (
    "src/query_adaptive_rag/evaluation/day3_heldout_m4v3.py",
    "src/query_adaptive_rag/evaluation/day3.py",
    "src/query_adaptive_rag/evaluation/protocol.py",
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integrity_hashes(root: Path, config_file: Path,
                     extra_sources: tuple[Path, ...] = (),
                     source_paths: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Hash the effective config and implementation sources for the manifest."""
    source_paths = HASHED_SOURCE_PATHS if source_paths is None else source_paths
    source_hashes = {relative: _file_sha256(root / relative) for relative in source_paths}
    for source in extra_sources:
        source_hashes[str(source.relative_to(root))] = _file_sha256(source)
    config_hash = _file_sha256(config_file)
    source_hashes["config"] = config_hash
    return {"configuration_path": str(config_file), "configuration_sha256": config_hash,
            "source_sha256": source_hashes}


def _ordered_id_hash(pairs: list[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()


def _manifest_eval_ids(manifest: Mapping[str, Any], dataset: str) -> list[str]:
    return [str(value) for value in manifest.get("dataset_splits", {}).get(dataset, {}).get("evaluation_ids", [])]


def validate_heldout_split(
    split_manifest: Mapping[str, Any],
    original_manifest: Mapping[str, Any],
    corrected_manifest: Mapping[str, Any],
    development_manifest: Mapping[str, Any],
    expected_hash: str = EXPECTED_SPLIT_HASH,
) -> tuple[list[tuple[str, str]], str]:
    """Require the exact original/corrected 400-ID list and disjoint dev IDs."""
    pairs: list[tuple[str, str]] = []
    for dataset in ("squad", "hotpotqa"):
        split = split_manifest.get(dataset, {})
        evaluation_ids = [str(value) for value in split.get("evaluation_ids", [])]
        if len(evaluation_ids) != 200 or len(set(evaluation_ids)) != 200:
            raise ValueError(f"{dataset} must have exactly 200 unique held-out query IDs")
        if evaluation_ids != _manifest_eval_ids(original_manifest, dataset):
            raise ValueError(f"{dataset} held-out IDs/order differ from original Day 3 manifest")
        if evaluation_ids != _manifest_eval_ids(corrected_manifest, dataset):
            raise ValueError(f"{dataset} held-out IDs/order differ from corrected Day 3 manifest")
        dev_split = development_manifest.get("dataset_splits", {}).get(dataset, {})
        development_ids = [str(value) for value in dev_split.get("development_ids", [])]
        if len(development_ids) != 50 or len(set(development_ids)) != 50:
            raise ValueError(f"{dataset} development reference must contain 50 unique IDs")
        generated_development_ids = [str(value) for value in split.get("development_ids", [])]
        if generated_development_ids != development_ids:
            raise ValueError(f"{dataset} generated development IDs differ from saved development IDs")
        for label, reference in (("original", original_manifest), ("corrected", corrected_manifest)):
            reference_dev = [str(value) for value in reference.get("dataset_splits", {}).get(dataset, {}).get("development_ids", [])]
            if development_ids != reference_dev:
                raise ValueError(f"{dataset} development IDs differ from the {label} Day 3 manifest")
        if set(evaluation_ids) & set(development_ids):
            raise ValueError(f"{dataset} held-out and development IDs overlap")
        pairs.extend((dataset, query_id) for query_id in evaluation_ids)
    if len(pairs) != 400 or len(set(pairs)) != 400:
        raise ValueError("held-out split must contain exactly 400 unique dataset/query pairs")
    digest = _ordered_id_hash(pairs)
    if digest != expected_hash:
        raise ValueError(f"ordered held-out ID hash mismatch: expected {expected_hash}, got {digest}")
    return pairs, digest


def validate_output_paths(root: Path, output_root: Path, processed_dir: Path, cache_dir: Path) -> None:
    """Refuse existing destinations and any baseline/ancestor/descendant path."""
    root = root.resolve()
    output_root = output_root.resolve()
    processed_dir = processed_dir.resolve()
    cache_dir = cache_dir.resolve()
    expected = (root / "results" / "day3_heldout_m4v3_alias_fix").resolve()
    if output_root != expected:
        raise ValueError(f"held-out outputs must use the isolated destination {expected}")
    for path in (processed_dir, cache_dir):
        if path != output_root and output_root not in path.parents:
            raise ValueError(f"all generated files must stay beneath {output_root}: {path}")
    baseline_paths = [
        root / "results" / name for name in (
            "day3", "day3_corrected", "day3_dev_m4v2", "day3_dev_m4v3",
            "day3_dev_m4v3_fact", "day3_dev_m4v3_fact_alias_fix",
        )
    ]
    baseline_paths += [root / "data" / "processed" / name for name in (
        "day3", "day3_corrected", "day3_dev_m4v2", "day3_dev_m4v3",
        "day3_dev_m4v3_fact", "day3_dev_m4v3_fact_alias_fix",
    )]
    for baseline in baseline_paths:
        baseline = baseline.resolve()
        if output_root == baseline or output_root in baseline.parents or baseline in output_root.parents:
            raise ValueError(f"isolated output overlaps an existing baseline path: {baseline}")
    if output_root.exists():
        raise FileExistsError(f"refusing to write: held-out output directory already exists: {output_root}")


def validate_frozen_settings(
    config: Mapping[str, Any], corrected_manifest: Mapping[str, Any],
    m4v2_reference: Mapping[str, Any], m4v3_reference: Mapping[str, Any],
) -> None:
    """Compare shared protocol and expansion settings to the approved configs."""
    if tuple(config.get("methods", ())) != METHODS:
        raise ValueError(f"configuration methods must be exactly {list(METHODS)}")
    baseline = corrected_manifest.get("effective_config", {})
    shared_keys = (
        "seed", "model_name", "tokenizer_name", "device", "embedding_max_sequence_length",
        "encoder_safe_m2", "dataset_source_split", "candidate_pool_size_per_dataset",
        "development_size_per_dataset", "evaluation_size_per_dataset", "top_k", "token_budget",
        "fixed_overlap_tokens", "semantic_similarity_threshold", "semantic_min_tokens",
        "semantic_max_tokens", "minimum_neighbor_similarity", "bootstrap_repeats",
        "expansion_rules", "datasets",
    )
    for key in shared_keys:
        if config.get(key) != baseline.get(key):
            raise ValueError(f"held-out setting {key!r} differs from corrected Day 3 protocol")
    if not config.get("encoder_safe_m2"):
        raise ValueError("held-out protocol requires the encoder-safe 510-content-token M2")
    for label, actual, reference in (
        ("M4-v2", config, m4v2_reference),
        ("M4-v3", config, m4v3_reference),
    ):
        ref_rules = reference.get("expansion_rules")
        if actual.get("expansion_rules") != ref_rules:
            raise ValueError(f"{label} expansion caps differ from its approved development config")
    if float(config["minimum_neighbor_similarity"]) != float(m4v2_reference["minimum_neighbor_similarity"]):
        raise ValueError("M4-v2 relevance threshold differs from its development config")
    if float(config["m4_v2"]["redundancy_threshold"]) != float(m4v2_reference["redundancy_threshold"]):
        raise ValueError("M4-v2 redundancy threshold differs from its development config")
    if config.get("m4_v3") != m4v3_reference.get("m4_v3"):
        raise ValueError("M4-v3 fact-expansion settings differ from its development config")


def _validate_runtime_record(record: Mapping[str, Any]) -> None:
    """Validate only non-label runtime fields before selection begins."""
    required = ("id", "dataset", "question", "documents")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"runtime query is missing fields {missing}")
    if record["dataset"] not in DATASET_COMPLEXITY:
        raise ValueError(f"unsupported dataset for query {record['id']}")
    if not str(record["question"]).strip() or not record["documents"]:
        raise ValueError(f"query {record['id']} has empty question or corpus context")


def runtime_query_views(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Strip answer/evidence fields before passing records to runtime logic."""
    return [{key: record[key] for key in ("id", "dataset", "question", "documents")}
            for record in records]


def _load_semantic_chunks(reuse_root: Path, corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = reuse_root / "chunks" / "M3_semantic.jsonl"
    documents = {str(row["doc_id"]): str(row["text"]) for row in corpus}
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            doc_id = str(chunk.get("doc_id", ""))
            if doc_id not in documents:
                raise ValueError(f"semantic chunk at line {line_number} is outside held-out corpus")
            if chunk.get("method") != "semantic":
                raise ValueError(f"non-semantic chunk in reused M3 inventory at line {line_number}")
            start, end = int(chunk["start"]), int(chunk["end"])
            if start < 0 or end < start or documents[doc_id][start:end] != str(chunk.get("text", "")):
                raise ValueError(f"semantic chunk source offsets mismatch at line {line_number}")
            chunks.append(chunk)
    ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    if not chunks or len(ids) != len(set(ids)):
        raise ValueError("reused M3 semantic inventory is empty or has duplicate chunk IDs")
    if {str(chunk["doc_id"]) for chunk in chunks} != set(documents):
        raise ValueError("reused M3 semantic inventory does not cover the held-out corpus")
    return chunks


def _query_embedding_hash(search: SearchResponse) -> str:
    return hashlib.sha256(np.asarray(search.query_embedding, dtype=np.float32).tobytes()).hexdigest()


def _selection_row(
    method: str,
    search: SearchResponse,
    selected_chunks: list[dict[str, Any]],
    context_text: str,
    token_count: int,
    token_budget: int,
    tokenizer: OffsetTokenizer,
    expansion_latency_ms: float = 0.0,
    expanded_ids: list[str] | None = None,
    similarity_scores: Mapping[str, float] | None = None,
    candidate_decisions: list[dict[str, Any]] | None = None,
    budget_excluded_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    rendered_text, rendered_count = assemble_context(selected_chunks, tokenizer)
    if rendered_text != context_text or rendered_count != token_count:
        raise RuntimeError(f"{method} context text/chunks/token count are inconsistent")
    expected_count = count_tokens(context_text, tokenizer)
    if expected_count != token_count:
        raise RuntimeError(f"{method} context token count mismatch: {token_count} vs {expected_count}")
    if token_count > token_budget:
        raise TokenBudgetError(f"{method} context {token_count} exceeds budget {token_budget}")
    return {
        "method": method,
        "search": search,
        "ranked_chunks": [hit.chunk for hit in search.hits],
        "seed_chunk_ids": [hit.chunk["chunk_id"] for hit in search.hits],
        "seed_ranks": [hit.rank for hit in search.hits],
        "seed_scores": {hit.chunk["chunk_id"]: float(hit.score) for hit in search.hits},
        "query_embedding_sha256": _query_embedding_hash(search),
        "selected_chunks": selected_chunks,
        "context_text": context_text,
        "context_token_count": token_count,
        "token_budget": token_budget,
        "expanded_chunk_ids": expanded_ids or [],
        "similarity_scores": dict(similarity_scores or {}),
        "expansion_latency_ms": float(expansion_latency_ms),
        "candidate_decisions": candidate_decisions or [],
        "budget_excluded_chunk_ids": budget_excluded_chunk_ids or [],
    }


def select_contexts_for_query(
    query: str,
    complexity: str,
    method_chunks: Mapping[str, list[dict[str, Any]]],
    retrievers: Mapping[str, DenseRetriever],
    tokenizer: OffsetTokenizer,
    rules: Mapping[str, ExpansionRule],
    token_budget: int,
    min_neighbor_similarity: float,
    redundancy_threshold: float,
    v3_config: ExpansionV3Config,
    top_k: int,
    *,
    chunk_embeddings: Mapping[str, np.ndarray] | None = None,
    chunk_by_id: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select all six method contexts; expansion variants share one M3 search."""
    searches = {method: retrievers[method].retrieve(query, top_k=top_k) for method in (M1, M2, M3)}
    m3_search = searches[M3]
    result: dict[str, dict[str, Any]] = {}
    for method in (M1, M2, M3):
        ranked = [hit.chunk for hit in searches[method].hits]
        selected, text, count, excluded = _budget_context(ranked, tokenizer, token_budget)
        result[method] = _selection_row(
            method, searches[method], selected, text, count, token_budget, tokenizer,
            budget_excluded_chunk_ids=excluded,
        )

    m3_chunks = method_chunks[M3]
    m3_chunk_by_id = chunk_by_id or {chunk["chunk_id"]: chunk for chunk in m3_chunks}
    m3_retriever = retrievers[M3]
    if chunk_embeddings is None:
        chunk_embeddings = chunk_embedding_map_from_retriever(m3_retriever)
    common = dict(search=m3_search, chunks=m3_chunks, tokenizer=tokenizer, rules=rules,
                  token_budget=token_budget)

    m4 = expand_context(
        **common, complexity=complexity, score_chunk=m3_retriever.score,
        min_similarity=min_neighbor_similarity,
    )
    if m4.seed_chunk_ids != result[M3]["seed_chunk_ids"]:
        raise RuntimeError("M4 changed M3 seed IDs or ordering")
    m4_ids = m4.seed_chunk_ids + m4.expanded_chunk_ids
    m4_chunks = [m3_chunk_by_id[cid] for cid in m4_ids]
    seed_set = set(m4.seed_chunk_ids)
    m4_decisions = []
    for cid, score in sorted(m4.similarity_scores.items()):
        if cid in seed_set:
            continue
        reason = ("accepted" if cid in m4.expanded_chunk_ids else
                  "token_budget_exceeded" if cid in m4.budget_rejected_chunk_ids else
                  "deduplicated_text" if cid in m4.deduplicated_chunk_ids else
                  "not_added_by_existing_policy")
        m4_decisions.append({"candidate_chunk_id": cid, "query_similarity": float(score),
                             "accepted": cid in m4.expanded_chunk_ids, "reason": reason})
    result[M4] = _selection_row(
        M4, m3_search, m4_chunks, m4.context_text, m4.context_token_count, token_budget, tokenizer,
        m4.latency_ms, m4.expanded_chunk_ids, m4.similarity_scores, m4_decisions,
    )

    m4v2 = expand_context_v2(
        **common, complexity=complexity, chunk_embeddings=chunk_embeddings,
        score_chunk=m3_retriever.score, min_similarity=min_neighbor_similarity,
        redundancy_threshold=redundancy_threshold,
    )
    if m4v2.seed_chunk_ids != result[M3]["seed_chunk_ids"]:
        raise RuntimeError("M4-v2 changed M3 seed IDs or ordering")
    m4v2_ids = m4v2.seed_chunk_ids + m4v2.expanded_chunk_ids
    m4v2_chunks = [m3_chunk_by_id[cid] for cid in m4v2_ids]
    m4v2_decisions = []
    for cid, score in sorted(m4v2.similarity_scores.items()):
        if cid in seed_set:
            continue
        reason = ("accepted" if cid in m4v2.expanded_chunk_ids else
                  "below_relevance_floor" if cid in m4v2.similarity_rejected_chunk_ids else
                  "redundant" if cid in m4v2.redundancy_rejected_chunk_ids else
                  "token_budget_exceeded" if cid in m4v2.budget_rejected_chunk_ids else
                  "deduplicated_text" if cid in m4v2.deduplicated_chunk_ids else
                  "not_added_by_existing_policy")
        m4v2_decisions.append({
            "candidate_chunk_id": cid, "query_similarity": float(score),
            "redundancy_similarity": m4v2.redundancy_scores.get(cid),
            "accepted": cid in m4v2.expanded_chunk_ids, "reason": reason,
        })
    result[M4V2] = _selection_row(
        M4V2, m3_search, m4v2_chunks, m4v2.context_text, m4v2.context_token_count, token_budget, tokenizer,
        m4v2.latency_ms, m4v2.expanded_chunk_ids, m4v2.similarity_scores, m4v2_decisions,
    )

    m4v3 = expand_context_v3(
        m3_search, query, complexity, m3_chunks, chunk_embeddings, tokenizer,
        m3_retriever.score, rules, token_budget, v3_config,
    )
    if (m4v3.seed_chunk_ids != result[M3]["seed_chunk_ids"] or
            m4v3.seed_ranks != result[M3]["seed_ranks"]):
        raise RuntimeError("M4-v3 changed M3 seed IDs, ordering, or ranks")
    m4v3_ids = m4v3.seed_chunk_ids + m4v3.expanded_chunk_ids
    m4v3_chunks = [m3_chunk_by_id[cid] for cid in m4v3_ids]
    result[M4V3] = _selection_row(
        M4V3, m3_search, m4v3_chunks, m4v3.context_text, m4v3.context_token_count, token_budget, tokenizer,
        m4v3.latency_ms, m4v3.expanded_chunk_ids, m4v3.similarity_scores,
        m4v3.candidate_decisions,
    )

    shared_hash = result[M3]["query_embedding_sha256"]
    for method in (M4, M4V2, M4V3):
        if result[method]["search"] is not m3_search or result[method]["query_embedding_sha256"] != shared_hash:
            raise RuntimeError(f"{method} did not receive the exact M3 search response")
        if (result[method]["seed_chunk_ids"] != result[M3]["seed_chunk_ids"] or
                result[method]["seed_ranks"] != result[M3]["seed_ranks"] or
                result[method]["seed_scores"] != result[M3]["seed_scores"]):
            raise RuntimeError(f"{method} M3 seed identity/ranking/scores changed")
    if set(result) != set(METHODS):
        raise RuntimeError("not all six evaluation methods produced selected contexts")
    return result


def select_all_contexts(
    records: list[dict[str, Any]],
    method_chunks: Mapping[str, list[dict[str, Any]]],
    retrievers: Mapping[str, DenseRetriever],
    tokenizer: OffsetTokenizer,
    rules: Mapping[str, ExpansionRule],
    token_budget: int,
    min_neighbor_similarity: float,
    redundancy_threshold: float,
    v3_config: ExpansionV3Config,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select contexts for every query before any gold evidence is accessed."""
    pending: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    for record in records:
        _validate_runtime_record(record)
        query = str(record["question"])
        classification = classify_query(query)
        complexity = classification["complexity"]
        class_counts[complexity] = class_counts.get(complexity, 0) + 1
        selections = select_contexts_for_query(
            query, complexity, method_chunks, retrievers, tokenizer, rules, token_budget,
            min_neighbor_similarity, redundancy_threshold, v3_config, top_k,
        )
        pending.append({
            "query_id": str(record["id"]), "dataset": str(record["dataset"]),
            "query_text": query, "classification": classification, "selections": selections,
        })
    return pending, class_counts


def _validate_evidence_for_metrics(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate gold spans only after context selection has finished."""
    evidence = deduplicate_evidence(record.get("gold_evidence", []))
    if not evidence:
        raise ValueError(f"query {record.get('id')} has no gold evidence; refusing to skip it")
    documents = {str(doc["doc_id"]): str(doc["text"]) for doc in record["documents"]}
    for span in evidence:
        text = documents.get(str(span.get("doc_id", span.get("document_id", ""))))
        if text is None or not (0 <= int(span["start"]) < int(span["end"]) <= len(text)):
            raise ValueError(f"query {record.get('id')} has invalid evaluation evidence span")
    return evidence


def score_selected_contexts(
    pending: list[dict[str, Any]], records: list[dict[str, Any]], token_budget: int,
) -> list[dict[str, Any]]:
    """Calculate evidence metrics after all query contexts have been selected."""
    records_by_key = {(str(record["dataset"]), str(record["id"])): record for record in records}
    output: list[dict[str, Any]] = []
    for item in pending:
        key = (item["dataset"], item["query_id"])
        record = records_by_key[key]
        evidence = _validate_evidence_for_metrics(record)
        for method in METHODS:
            selection = item["selections"][method]
            if selection["context_token_count"] > token_budget:
                raise TokenBudgetError(f"{method} exceeded the hard context budget")
            metrics = calculate_metrics(
                selection["ranked_chunks"],
                context_source_chunks(selection["selected_chunks"]),
                evidence,
            )
            output.append({
                "query_id": item["query_id"],
                "dataset": item["dataset"],
                "evaluation_complexity": DATASET_COMPLEXITY[item["dataset"]],
                "predicted_complexity": item["classification"]["complexity"],
                "classification_features": item["classification"]["features"],
                "method": method,
                "query_text": item["query_text"],
                "seed_chunk_ids": selection["seed_chunk_ids"],
                "seed_ranks": selection["seed_ranks"],
                "seed_scores": selection["seed_scores"],
                "query_embedding_sha256": selection["query_embedding_sha256"],
                "expanded_chunk_ids": selection["expanded_chunk_ids"],
                "similarity_scores": selection["similarity_scores"],
                "final_context_text": selection["context_text"],
                **metrics,
                "context_token_count": selection["context_token_count"],
                "token_budget": selection["token_budget"],
                "retrieval_latency_ms": selection["search"].latency_ms,
                "expansion_latency_ms": selection["expansion_latency_ms"],
                "candidate_neighbors_considered": len(selection["candidate_decisions"]),
                "neighbors_accepted": len(selection["expanded_chunk_ids"]) if method in EXPANSION_METHODS else 0,
                "neighbors_rejected": (sum(not bool(row.get("accepted")) for row in selection["candidate_decisions"])
                                       if method in EXPANSION_METHODS else 0),
                "budget_excluded_chunk_ids": selection["budget_excluded_chunk_ids"],
            })
    return output


def _method_protocol(model_name: str, top_k: int, token_budget: int,
                     m2_content_tokens: int, special_tokens: int,
                     redundancy_threshold: float, v3_config: ExpansionV3Config) -> dict[str, dict[str, Any]]:
    base = method_protocol(model_name, top_k, token_budget, 32, 512, m2_content_tokens, special_tokens)
    base.pop("M2_fixed_512", None)
    base[M2] = {
        "embedding_model": model_name, "embedding_max_sequence_length": 512,
        "top_k": top_k, "token_budget": token_budget,
        "chunking": "fixed", "chunk_size": m2_content_tokens,
        "chunk_size_unit": "content_tokens", "overlap_tokens": 32,
        "tokenizer_special_token_count": special_tokens,
    }
    base[M4V2] = {**base[M4], "expansion": "redundancy-aware", "redundancy_threshold": redundancy_threshold}
    base[M4V3] = {
        **base[M4], "expansion": "fact-level incremental", "alias_resolution": "corrected Gold Center",
        "config": vars(v3_config),
    }
    return {key: base[key] for key in METHODS}


def _report_text(rows: list[dict[str, Any]], manifest: Mapping[str, Any]) -> str:
    aggregate = _aggregate(rows)
    all_rows = {row["method"]: row for row in aggregate if row["dataset"] == "all"}
    lines = [
        "# Six-Method Held-Out Evaluation",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Held-out queries: {manifest['evaluation_query_count']} (200 SQuAD, 200 HotpotQA)",
        f"- Ordered query ID hash: `{manifest['heldout_query_ids_sha256']}`",
        f"- M3 semantic chunks: {manifest['semantic_chunk_count']}",
        "",
        "## Aggregate results",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Context precision | Mean context tokens | Expansion latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = all_rows[method]
        lines.append(
            f"| {method} | {row['mean_recall_at_1']:.4f} | {row['mean_recall_at_3']:.4f} | "
            f"{row['mean_recall_at_5']:.4f} | {row['mean_mrr']:.4f} | "
            f"{row['mean_context_precision']:.4f} | {row['mean_context_token_count']:.2f} | "
            f"{row['mean_expansion_latency_ms']:.3f} |"
        )
    lines.extend(["", "## Dataset-level results", "", "| Method | Dataset | Queries | Context precision | Mean context tokens | Expansion latency (ms) |", "|---|---|---:|---:|---:|---:|"])
    for row in aggregate:
        if row["dataset"] == "all":
            continue
        lines.append(
            f"| {row['method']} | {row['dataset']} | {row['query_count']} | "
            f"{row['mean_context_precision']:.4f} | {row['mean_context_token_count']:.2f} | "
            f"{row['mean_expansion_latency_ms']:.3f} |"
        )
    lines.extend([
        "", "## Runtime and evidence controls", "",
        "M4, M4-v2, and M4-v3 used the same M3 search response for each query. "
        "Gold evidence was passed to metric calculation only after contexts for all held-out queries were selected. "
        "No answer-generation metric was substituted for evidence retrieval.",
        "", "## Candidate decisions", "",
        "Per-query, per-method candidate decisions and acceptance/rejection reasons are in `candidate_decisions.jsonl`.",
        "", "All generated contexts were checked against the configured hard token budget before metric calculation.",
    ])
    return "\n".join(lines) + "\n"


def _write_six_method_plots(aggregate_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Write compact SVG comparisons without changing the four-method plotter."""
    from html import escape

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {row["method"]: row for row in aggregate_rows if row["dataset"] == "all"}
    labels = ("Fixed 256", "Fixed 510", "Semantic", "M4", "M4-v2", "M4-v3")
    colors = ("#3569a8", "#54a24b", "#f28e2b", "#b279a2", "#e45756", "#72b7b2")
    outputs = []
    for metric, title, filename in (
        ("mean_recall_at_5", "Held-out evidence Recall@5", "recall_at_5.svg"),
        ("mean_context_precision", "Held-out context precision", "context_precision.svg"),
        ("mean_context_token_count", "Held-out mean context tokens", "context_tokens.svg"),
    ):
        values = [float(all_rows[method][metric]) for method in METHODS]
        maximum = max(values, default=0.0) or 1.0
        width, height, left, top, chart_height = 900, 470, 50, 75, 310
        slot = 800 / len(METHODS)
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 f'<text x="450" y="35" text-anchor="middle" font-size="20" font-family="Arial">{escape(title)}</text>',
                 f'<line x1="{left}" y1="{top+chart_height}" x2="{left+800}" y2="{top+chart_height}" stroke="#555"/>']
        for index, (label, method, value) in enumerate(zip(labels, METHODS, values)):
            bar_height = chart_height * value / maximum
            bar_width = slot * 0.62
            x = left + index * slot + slot * 0.19
            y = top + chart_height - bar_height
            parts.extend([
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[index]}"/>',
                f'<text x="{x+bar_width/2:.1f}" y="{max(y-7,top):.1f}" text-anchor="middle" font-size="12" font-family="Arial">{value:.3f}</text>',
                f'<text x="{x+bar_width/2:.1f}" y="{top+chart_height+25}" text-anchor="middle" font-size="11" font-family="Arial">{escape(label)}</text>',
            ])
        parts.append("</svg>")
        path = output_dir / filename
        path.write_text("\n".join(parts), encoding="utf-8")
        outputs.append(path)
    return outputs


def run(config_path: str | Path, executed_command: str | None = None) -> tuple[Path, Path]:
    config_file = Path(config_path).resolve()
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    seed = int(raw.get("seed", 42))
    set_random_seed(seed)

    results_dir = (root / raw["results_output_dir"]).resolve()
    processed_dir = (root / raw["processed_output_dir"]).resolve()
    cache_dir = (root / raw["embedding_cache_dir"]).resolve()
    validate_output_paths(root, results_dir, processed_dir, cache_dir)

    original_path = (root / raw["original_reference_manifest"]).resolve()
    corrected_path = (root / raw["reference_manifest"]).resolve()
    development_path = (root / raw["development_reference_manifest"]).resolve()
    original_manifest = json.loads(original_path.read_text(encoding="utf-8"))
    corrected_manifest = json.loads(corrected_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_path.read_text(encoding="utf-8"))
    m4v2_config_path = (root / raw["m4_v2_reference_config"]).resolve()
    m4v3_config_path = (root / raw["m4_v3_reference_config"]).resolve()
    m4v2_reference = yaml.safe_load(m4v2_config_path.read_text(encoding="utf-8"))
    m4v3_reference = yaml.safe_load(m4v3_config_path.read_text(encoding="utf-8"))
    validate_frozen_settings(raw, corrected_manifest, m4v2_reference, m4v3_reference)

    evaluation_records, split_manifest = _heldout_records(raw, seed)
    configured_split_hash = str(raw.get("expected_heldout_query_ids_sha256", ""))
    if configured_split_hash != EXPECTED_SPLIT_HASH:
        raise ValueError("configuration held-out split hash differs from the frozen expected ID hash")
    query_pairs, split_hash = validate_heldout_split(
        split_manifest, original_manifest, corrected_manifest, development_manifest,
        configured_split_hash,
    )
    if len(evaluation_records) != 400:
        raise ValueError(f"expected 400 held-out records, got {len(evaluation_records)}")
    dataset_counts = {dataset: sum(record["dataset"] == dataset for record in evaluation_records)
                      for dataset in ("squad", "hotpotqa")}
    if dataset_counts != {"squad": 200, "hotpotqa": 200}:
        raise ValueError(f"expected 200 records per dataset, got {dataset_counts}")
    actual_pairs = [(str(record["dataset"]), str(record["id"])) for record in evaluation_records]
    if actual_pairs != query_pairs:
        raise ValueError("loaded held-out records differ from the verified ordered manifest IDs")
    for record in evaluation_records:
        _validate_runtime_record(record)
    corpus = _unique_corpus(evaluation_records)
    if not corpus:
        raise ValueError("held-out corpus is empty")

    model_name = str(raw["model_name"])
    tokenizer_name = str(raw.get("tokenizer_name", model_name))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("fast tokenizer required for reproducible token counts")
    encoder = SentenceTransformer(model_name, device=str(raw.get("device", "cpu")))
    max_sequence_length = int(raw.get("embedding_max_sequence_length", 512))
    model_config = encoder._first_module().auto_model.config
    model_position_limit = getattr(model_config, "max_position_embeddings", max_sequence_length)
    if max_sequence_length > model_position_limit:
        raise ValueError("configured embedding limit exceeds the model position limit")
    encoder.max_seq_length = max_sequence_length
    m2_content_tokens, special_token_count = encoder_content_token_limit(tokenizer, max_sequence_length)
    if m2_content_tokens != 510:
        raise ValueError(f"M2 protocol requires 510 content tokens; tokenizer yields {m2_content_tokens}")

    overlap = int(raw.get("fixed_overlap_tokens", 32))
    if overlap != 32:
        raise ValueError("the held-out frozen protocol requires 32-token overlap")
    m1_chunks: list[dict[str, Any]] = []
    m2_chunks: list[dict[str, Any]] = []
    m1_started = perf_counter()
    for document in corpus:
        m1_chunks.extend(fixed_size_chunks(document["text"], document["doc_id"], tokenizer, 256, overlap))
    m1_chunking_ms = (perf_counter() - m1_started) * 1000.0
    m2_started = perf_counter()
    for document in corpus:
        m2_chunks.extend(fixed_size_chunks(document["text"], document["doc_id"], tokenizer, m2_content_tokens, overlap))
    m2_chunking_ms = (perf_counter() - m2_started) * 1000.0
    reuse_root = (root / raw["reuse_semantic_chunks_from"]).resolve()
    semantic_started = perf_counter()
    m3_chunks = _load_semantic_chunks(reuse_root, corpus)
    semantic_inventory_validation_ms = (perf_counter() - semantic_started) * 1000.0
    chunking_times = {
        M1: m1_chunking_ms,
        M2: m2_chunking_ms,
        M3: 0.0,
    }

    embedding_audits = {
        method: audit_chunk_embedding_lengths(chunks, tokenizer, max_sequence_length)
        for method, chunks in ((M1, m1_chunks), (M2, m2_chunks), (M3, m3_chunks))
    }
    method_chunks = {M1: m1_chunks, M2: m2_chunks, M3: m3_chunks}
    retrievers: dict[str, DenseRetriever] = {}
    index_build_times: dict[str, float] = {}
    for method in (M1, M2, M3):
        retriever = DenseRetriever(encoder, model_name)
        started = perf_counter()
        retriever.build(method_chunks[method])
        index_build_times[method] = (perf_counter() - started) * 1000.0
        retrievers[method] = retriever

    rules = {name: ExpansionRule(int(values["max_depth"]), int(values["max_expansions"]))
             for name, values in raw["expansion_rules"].items()}
    top_k = int(raw.get("top_k", 5))
    budget = int(raw.get("token_budget", 3072))
    min_similarity = float(raw.get("minimum_neighbor_similarity", 0.20))
    redundancy_threshold = float(raw.get("m4_v2", {}).get("redundancy_threshold", 0.85))
    v3_config = ExpansionV3Config(**raw.get("m4_v3", {}))

    # Selection phase receives query text and documents only; gold fields remain unused.
    runtime_records = runtime_query_views(evaluation_records)
    pending, class_counts = select_all_contexts(
        runtime_records, method_chunks, retrievers, tokenizer, rules, budget,
        min_similarity, redundancy_threshold, v3_config, top_k,
    )
    if len(pending) != 400:
        raise RuntimeError("not all held-out queries completed context selection")

    # Only after every context-selection decision is complete are evidence spans accessed.
    result_rows = score_selected_contexts(pending, evaluation_records, budget)
    if len(result_rows) != 400 * len(METHODS):
        raise RuntimeError("one or more held-out query/method result rows are missing")

    split_pairs_hash = split_hash
    hash_record = integrity_hashes(root, config_file, (m4v2_config_path, m4v3_config_path))
    reuse_path = reuse_root / "chunks" / "M3_semantic.jsonl"
    reused_chunk_hash = _file_sha256(reuse_path)
    reused_chunk_id_hash = hashlib.sha256(
        "\n".join(chunk["chunk_id"] for chunk in m3_chunks).encode("utf-8")
    ).hexdigest()
    protocol = _method_protocol(
        model_name, top_k, budget, m2_content_tokens, special_token_count, redundancy_threshold, v3_config,
    )
    manifest = {
        "run_id": raw.get("run_id", "day3_heldout_m4v3_alias_fix_20261106"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed_command": executed_command or "programmatic call to day3_heldout_m4v3.run",
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        **hash_record,
        "expansion_reference_configs": {
            "m4_v2": str(m4v2_config_path), "m4_v3": str(m4v3_config_path),
        },
        "output_locations": {
            "results": str(results_dir), "processed": str(processed_dir), "cache": str(cache_dir),
        },
        "reference_manifests": {
            "original_day3": str(original_path), "corrected_day3": str(corrected_path),
            "development": str(development_path),
        },
        "seed": seed,
        "model_identifier": model_name,
        "tokenizer_identifier": tokenizer_name,
        "methods": list(METHODS),
        "method_protocol": protocol,
        "dataset_splits": split_manifest,
        "evaluation_query_count": len(evaluation_records),
        "completed_query_count": len(pending),
        "skipped_query_count": 0,
        "dataset_counts": dataset_counts,
        "development_disjointness_verified": True,
        "heldout_query_ids_sha256": split_pairs_hash,
        "unique_corpus_document_count": len(corpus),
        "semantic_chunk_count": len(m3_chunks),
        "semantic_chunks_reused_from": str(reuse_path),
        "semantic_chunk_file_sha256": reused_chunk_hash,
        "semantic_chunk_ids_sha256": reused_chunk_id_hash,
        "embedding_input_audit": embedding_audits,
        "chunking_latency_ms": chunking_times,
        "semantic_inventory_validation_ms": semantic_inventory_validation_ms,
        "index_build_latency_ms": index_build_times,
        "m3_search_shared_by": [M3, M4, M4V2, M4V3],
        "m3_seed_ids_ranks_scores_and_query_embedding_shared": True,
        "runtime_classifier_inputs": ["query text"],
        "runtime_retrieval_inputs": ["query text", "indexed chunks"],
        "runtime_expansion_inputs": ["query text", "predicted complexity", "M3 search", "semantic chunks", "chunk embeddings"],
        "gold_evidence_usage": "evaluation metrics only after all contexts have been selected",
        "gold_answers_used_at_runtime": False,
        "heldout_answers_or_evidence_used_to_tune": False,
        "classifier_prediction_counts": class_counts,
        "result_rows": len(result_rows),
        "all_contexts_within_budget": all(int(row["context_token_count"]) <= int(row["token_budget"])
                                           for row in result_rows),
        "maximum_context_tokens": max(int(row["context_token_count"]) for row in result_rows),
        "aggregate_rows": len(_aggregate(result_rows)),
        "outputs": [],
    }

    # No project output is written until the split, selection, metrics and budgets validate.
    results_dir.mkdir(parents=True, exist_ok=False)
    processed_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=False)
    for method, chunks in method_chunks.items():
        _write_jsonl(processed_dir / f"{method}_chunks.jsonl", chunks)
        retrievers[method].save(processed_dir / f"{method}.faiss")
        _write_jsonl(processed_dir / f"{method}_rows.jsonl", [
            {"index_row": index, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]}
            for index, chunk in enumerate(retrievers[method].chunks)
        ])
    _write_jsonl(results_dir / "per_query.jsonl", result_rows)
    csv_rows = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                 for key, value in row.items()} for row in result_rows]
    with (results_dir / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    aggregate_rows = _aggregate(result_rows)
    _write_csv(results_dir / "aggregate_metrics.csv", aggregate_rows)

    paired_rows: list[dict[str, Any]] = []
    for method in METHODS:
        if method == M3:
            continue
        paired_rows.extend(paired_bootstrap_comparisons(
            result_rows, method_a=M3, method_b=method,
            repeats=int(raw.get("bootstrap_repeats", 1000)), seed=seed,
        ))
    _write_csv(results_dir / "paired_bootstrap_ci.csv", paired_rows)
    decision_rows = []
    for item in pending:
        for method in METHODS:
            selection = item["selections"][method]
            decision_rows.append({
                "query_id": item["query_id"], "dataset": item["dataset"], "method": method,
                "seed_chunk_ids": selection["seed_chunk_ids"],
                "seed_ranks": selection["seed_ranks"],
                "query_embedding_sha256": selection["query_embedding_sha256"],
                "decisions": selection["candidate_decisions"],
                "expanded_chunk_ids": selection["expanded_chunk_ids"],
                "budget_excluded_chunk_ids": selection["budget_excluded_chunk_ids"],
                "context_token_count": selection["context_token_count"],
                "token_budget": budget,
            })
    _write_jsonl(results_dir / "candidate_decisions.jsonl", decision_rows)
    plots = _write_six_method_plots(aggregate_rows, results_dir / "plots")
    report_path = results_dir / "heldout_evaluation_report.md"
    report_path.write_text(_report_text(result_rows, manifest), encoding="utf-8")
    manifest["outputs"] = [
        "per_query.csv", "per_query.jsonl", "aggregate_metrics.csv", "paired_bootstrap_ci.csv",
        "candidate_decisions.jsonl", "heldout_evaluation_report.md", "experiment_manifest.json",
        "processed/", "cache/", *[str(Path("plots") / path.name) for path in plots],
    ]
    (results_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return processed_dir, results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated six-method held-out M4-v3 evaluation")
    parser.add_argument("--config", default="configs/day3_heldout_m4v3_alias_fix.yaml")
    args = parser.parse_args()
    command = subprocess.list2cmdline([
        sys.executable, "-m", "query_adaptive_rag.evaluation.day3_heldout_m4v3", *sys.argv[1:],
    ])
    processed, results = run(args.config, command)
    print(f"Six-method held-out evaluation complete: {processed}; {results}")


if __name__ == "__main__":
    main()
