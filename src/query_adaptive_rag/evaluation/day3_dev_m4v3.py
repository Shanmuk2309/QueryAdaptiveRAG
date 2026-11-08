"""Isolated development-only evaluation pipeline for M3 and M4-v3."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import yaml

from ..chunking.semantic import semantic_chunks
from ..chunking.text_utils import OffsetTokenizer
from ..classification.complexity import classify_query
from ..config import set_random_seed
from ..data.loaders import _hotpot_records, _squad_records
from ..expansion.context_expander import ExpansionRule, assemble_context
from ..expansion.context_expander_v3 import ExpansionV3Config, expand_context_v3
from ..retrieval.dense import DenseRetriever, SearchResponse
from .context import context_source_chunks
from .day3 import _unique_corpus, _validate_query_record
from .day3_dev_m4v2 import (
    _dataset_id, _development_and_heldout_ids, _select_development_rows,
    _write_csv, _write_jsonl,
)
from .evidence import deduplicate_evidence
from .metrics import calculate_metrics


METHODS = ("M3_no_expansion", "M4_v3_incremental")
DATASET_COMPLEXITY = {"squad": "single_fact", "hotpotqa": "multi_hop"}
METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "context_precision")


def _load_reused_semantic_chunks(path: Path, corpus: list[dict[str, Any]],
                                expected_sha256: str | None = None) -> list[dict[str, Any]]:
    """Load exact baseline chunks after verifying they belong to this corpus."""
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and actual_sha256.casefold() != expected_sha256.casefold():
        raise ValueError(f"reused semantic chunk file hash mismatch: expected {expected_sha256}, got {actual_sha256}")
    documents = {str(row["doc_id"]): str(row["text"]) for row in corpus}
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            doc_id = str(chunk.get("doc_id", ""))
            if doc_id not in documents:
                raise ValueError(f"reused semantic chunk references an unexpected document at line {line_number}: {doc_id}")
            if chunk.get("method") != "semantic":
                raise ValueError(f"expected semantic chunks; line {line_number} has method {chunk.get('method')!r}")
            start, end = int(chunk["start"]), int(chunk["end"])
            if start < 0 or end < start or documents[doc_id][start:end] != str(chunk.get("text", "")):
                raise ValueError(f"reused semantic chunk text/offset mismatch at line {line_number}: {chunk.get('chunk_id')}")
            chunks.append(chunk)
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    if not chunks or len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("reused semantic chunk file is empty or contains duplicate chunk IDs")
    if {str(chunk["doc_id"]) for chunk in chunks} != set(documents):
        raise ValueError("reused semantic chunks do not cover exactly the development corpus documents")
    return chunks


def _load_dev(config: dict[str, Any], reference_manifest: dict[str, Any]):
    """Reconstruct the saved development IDs and normalize development rows only."""
    from datasets import load_dataset

    records: list[dict[str, Any]] = []
    split_info: dict[str, Any] = {}
    seed = int(config.get("seed", 42))
    for offset, key in enumerate(("squad", "hotpotqa")):
        settings = config["datasets"][key]
        pool_size = int(config["candidate_pool_size_per_dataset"])
        dev_size = int(config["development_size_per_dataset"])
        heldout_size = int(config["heldout_size_per_dataset"])
        dataset_name = settings["dataset_name"]
        dataset_config = settings.get("dataset_config")
        source_split = settings.get("split", "validation")
        raw = (load_dataset(dataset_name, dataset_config, split=source_split)
               if dataset_config else load_dataset(dataset_name, split=source_split))
        sampled = raw.shuffle(seed=seed + offset).select(range(min(pool_size, len(raw))))
        candidate_ids = [_dataset_id(dict(row), key) for row in sampled]
        if len(candidate_ids) < dev_size + heldout_size:
            raise ValueError(f"{key} candidate pool is smaller than dev + heldout split")
        dev_ids, heldout_ids = _development_and_heldout_ids(
            candidate_ids, dev_size, heldout_size, seed + 100 + offset,
        )
        reference = reference_manifest.get("dataset_splits", {}).get(key, {})
        if dev_ids != reference.get("development_ids"):
            raise ValueError(f"{key} development IDs differ from the original Day 3 split")
        if heldout_ids != reference.get("evaluation_ids"):
            raise ValueError(f"{key} held-out split IDs differ from the original Day 3 split")
        selected = _select_development_rows(sampled, key, dev_ids)
        normalized = _squad_records(selected) if key == "squad" else _hotpot_records(selected)
        if [row["id"] for row in normalized] != dev_ids:
            raise ValueError(f"{key} development normalization changed IDs/order")
        records.extend(normalized)
        split_info[key] = {
            "dataset_name": dataset_name, "dataset_config": dataset_config,
            "source_split": source_split, "candidate_pool_count": len(candidate_ids),
            "development_count": len(dev_ids), "development_ids": dev_ids,
            "development_matches_reference_manifest": True,
            "heldout_ids_used_for_evaluation": False,
            "heldout_payloads_normalized_or_used": False,
        }
    return records, split_info


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        for dataset in ("squad", "hotpotqa", "all"):
            group = [row for row in rows if row["method"] == method and
                     (dataset == "all" or row["dataset"] == dataset)]
            if not group:
                continue
            item: dict[str, Any] = {
                "method": method, "dataset": dataset,
                "query_count": len(group),
                "evaluation_complexity": "mixed" if dataset == "all" else DATASET_COMPLEXITY[dataset],
            }
            for metric in (*METRICS, "context_token_count", "retrieval_latency_ms", "expansion_latency_ms"):
                item[f"mean_{metric}"] = float(np.mean([float(row[metric]) for row in group]))
            item["total_candidates"] = sum(int(row["candidate_neighbors_considered"]) for row in group)
            item["total_accepted"] = sum(int(row["neighbors_accepted"]) for row in group)
            item["total_rejected"] = sum(int(row["neighbors_rejected"]) for row in group)
            output.append(item)
    return output


def _metric_row(record: dict[str, Any], search: SearchResponse, context_chunks: list[dict[str, Any]],
                method: str, context_text: str, token_count: int, latency: float,
                expansion_result: Any | None, token_budget: int) -> dict[str, Any]:
    seeds = [hit.chunk for hit in search.hits]
    metrics = calculate_metrics(seeds, context_source_chunks(context_chunks), record["gold_evidence"])
    classification = classify_query(record["question"])
    result = {
        "query_id": record["id"], "dataset": record["dataset"],
        "evaluation_complexity": DATASET_COMPLEXITY[record["dataset"]],
        "predicted_complexity": classification["complexity"],
        "classification_features": classification["features"],
        "method": method, "query_text": record["question"],
        "seed_chunk_ids": [hit.chunk["chunk_id"] for hit in search.hits],
        "seed_ranks": [hit.rank for hit in search.hits],
        "expanded_chunk_ids": [] if expansion_result is None else expansion_result.expanded_chunk_ids,
        "similarity_scores": {hit.chunk["chunk_id"]: float(hit.score) for hit in search.hits},
        "final_context_text": context_text, **metrics,
        "context_token_count": token_count, "token_budget": token_budget,
        "retrieval_latency_ms": search.latency_ms, "expansion_latency_ms": latency,
        "candidate_neighbors_considered": 0 if expansion_result is None else len(expansion_result.candidate_decisions),
        "neighbors_accepted": 0 if expansion_result is None else len(expansion_result.expanded_chunk_ids),
        "neighbors_rejected": 0 if expansion_result is None else sum(
            not row["accepted"] for row in expansion_result.candidate_decisions
        ),
        "candidate_decisions": [] if expansion_result is None else expansion_result.candidate_decisions,
        "neighbor_similarity_scores": {} if expansion_result is None else expansion_result.similarity_scores,
        "neighbor_redundancy_scores": {} if expansion_result is None else expansion_result.redundancy_scores,
    }
    return result


def evaluate_query(record: dict[str, Any], search: SearchResponse, chunks: list[dict[str, Any]],
                   retriever: DenseRetriever, chunk_vectors: dict[str, np.ndarray],
                   tokenizer: OffsetTokenizer, rules: dict[str, ExpansionRule],
                   token_budget: int, v3_config: ExpansionV3Config) -> list[dict[str, Any]]:
    """Render M3 and M4-v3 from the same immutable M3 SearchResponse."""
    seeds = [hit.chunk for hit in search.hits]
    seed_ids = [hit.chunk["chunk_id"] for hit in search.hits]
    seed_context, seed_tokens = assemble_context(seeds, tokenizer)
    if seed_tokens > token_budget:
        raise ValueError(f"M3 seed context for {record['id']} exceeds the token budget")
    complexity = classify_query(record["question"])["complexity"]
    expansion = expand_context_v3(
        search, record["question"], complexity, chunks, chunk_vectors, tokenizer,
        retriever.score, rules, token_budget, v3_config,
    )
    if expansion.seed_chunk_ids != seed_ids or expansion.seed_ranks != [hit.rank for hit in search.hits]:
        raise RuntimeError("M4-v3 changed M3 seed IDs or ranks")
    expanded_chunks = [next(c for c in chunks if c["chunk_id"] == cid)
                       for cid in expansion.expanded_chunk_ids]
    return [
        _metric_row(record, search, seeds, "M3_no_expansion", seed_context, seed_tokens, 0.0, None, token_budget),
        _metric_row(record, search, [*seeds, *expanded_chunks], "M4_v3_incremental",
                    expansion.context_text, expansion.context_token_count, expansion.latency_ms,
                    expansion, token_budget),
    ]


def run(config_path: str | Path, executed_command: str | None = None) -> tuple[Path, Path]:
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    set_random_seed(int(config.get("seed", 42)))
    processed_dir = (root / config["processed_output_dir"]).resolve()
    results_dir = (root / config["results_output_dir"]).resolve()
    forbidden = [root / value for value in (
        "data/processed/day3", "data/processed/day3_corrected", "data/processed/day3_dev_m4v2",
        "results/day3", "results/day3_corrected", "results/day3_dev_m4v2",
    )]
    for destination in (processed_dir, results_dir):
        if any(destination == item.resolve() or item.resolve() in destination.parents for item in forbidden):
            raise ValueError(f"M4-v3 outputs must use isolated paths: {destination}")
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing M4-v3 output directory: {destination}")

    reference_path = (root / config["reference_manifest"]).resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    records, split_info = _load_dev(config, reference)
    if len(records) != 100:
        raise ValueError(f"expected 100 existing development records, got {len(records)}")
    if sum(row["dataset"] == "squad" for row in records) != 50 or sum(row["dataset"] == "hotpotqa" for row in records) != 50:
        raise ValueError("M4-v3 development split must have 50 SQuAD and 50 HotpotQA queries")
    for record in records:
        _validate_query_record(record)
    corpus = _unique_corpus(records)

    model_name = str(config["model_name"])
    tokenizer_name = str(config.get("tokenizer_name", model_name))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("fast tokenizer required")
    encoder = SentenceTransformer(model_name, device=str(config.get("device", "cpu")))
    model_limit = int(config.get("embedding_max_sequence_length", 512))
    encoder.max_seq_length = model_limit
    cache = (root / config.get("embedding_cache_dir", "data/cache/day3_dev_m4v3")).resolve()
    reused_chunks_path = config.get("semantic_chunks_input_path")
    chunk_started = perf_counter()
    if reused_chunks_path:
        chunks = _load_reused_semantic_chunks(
            (root / reused_chunks_path).resolve(), corpus, config.get("semantic_chunks_sha256"),
        )
    else:
        chunks: list[dict[str, Any]] = []
        for document in corpus:
            chunks.extend(semantic_chunks(
                document["text"], document["doc_id"], tokenizer, encoder,
                float(config.get("semantic_similarity_threshold", 0.45)),
                int(config.get("semantic_min_tokens", 80)), int(config.get("semantic_max_tokens", 400)),
                cache, model_name,
            ))
    chunking_ms = (perf_counter() - chunk_started) * 1000
    retriever = DenseRetriever(encoder, model_name)
    build_started = perf_counter()
    retriever.build(chunks)
    index_ms = (perf_counter() - build_started) * 1000
    chunk_vectors = {chunk["chunk_id"]: retriever.vectors[i] for i, chunk in enumerate(retriever.chunks)}
    rules = {key: ExpansionRule(int(value["max_depth"]), int(value["max_expansions"]))
             for key, value in config["expansion_rules"].items()}
    v3_config = ExpansionV3Config(**config.get("m4_v3", {}))
    top_k = int(config.get("top_k", 5))
    budget = int(config.get("token_budget", 3072))
    rows: list[dict[str, Any]] = []
    for record in records:
        response = retriever.retrieve(record["question"], top_k=top_k)
        rows.extend(evaluate_query(record, response, chunks, retriever, chunk_vectors,
                                   tokenizer, rules, budget, v3_config))
    if len(rows) != len(records) * 2:
        raise RuntimeError("M4-v3 evaluation omitted method rows")

    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(processed_dir / "M3_semantic_chunks.jsonl", chunks)
    retriever.save(processed_dir / "M3_semantic.faiss")
    _write_jsonl(processed_dir / "M3_semantic_rows.jsonl", [
        {"index_row": i, "chunk_id": row["chunk_id"], "doc_id": row["doc_id"]}
        for i, row in enumerate(retriever.chunks)
    ])
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(results_dir / "per_query.jsonl", rows)
    _write_csv(results_dir / "per_query.csv", [
        {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
         for key, value in row.items()} for row in rows
    ])
    aggregate = _aggregate(rows)
    _write_csv(results_dir / "aggregate_metrics.csv", aggregate)
    (results_dir / "candidate_decisions.jsonl").write_text(
        "".join(json.dumps({"query_id": row["query_id"], "method": row["method"],
                            "decisions": row["candidate_decisions"]}, ensure_ascii=False) + "\n"
                for row in rows if row["method"] == "M4_v3_incremental"), encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed_command": executed_command or "programmatic call to day3_dev_m4v3.run",
        "python_executable": sys.executable, "python_platform": platform.platform(),
        "dependency_versions": {name: metadata.version(name) for name in
                                ("numpy", "datasets", "sentence-transformers", "transformers", "torch", "faiss-cpu", "PyYAML")},
        "run_id": config.get("run_id", config.get("experiment_name", "day3_dev_m4v3")),
        "experiment_name": config.get("experiment_name", "day3_dev_m4v3"), "config": config,
        "protocol": {"methods": list(METHODS), "development_only": True,
                     "development_query_count": len(records), "same_m3_search_response_shared": True,
                     "same_m3_seed_ids_ranks_query_embedding": True,
                     "runtime_inputs": ["query text", "semantic chunks", "cached embeddings"],
                     "gold_evidence_used_for": "context precision/recall metrics only after selection",
                     "heldout_query_payloads_normalized_or_evaluated": False},
        "reference_manifest": str(reference_path), "dataset_splits": split_info,
        "semantic_chunk_count": len(chunks), "chunking_latency_ms": chunking_ms,
        "semantic_chunks_source": str((root / reused_chunks_path).resolve()) if reused_chunks_path else "regenerated",
        "semantic_chunks_file_sha256": hashlib.sha256(
            (root / reused_chunks_path).resolve().read_bytes()
        ).hexdigest() if reused_chunks_path else None,
        "semantic_chunk_ids_sha256": hashlib.sha256(
            "\n".join(str(chunk["chunk_id"]) for chunk in chunks).encode("utf-8")
        ).hexdigest(),
        "semantic_chunk_content_sha256": hashlib.sha256(json.dumps(
            chunks, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "index_build_latency_ms": index_ms, "result_rows": len(rows),
        "outputs": ["per_query.csv", "per_query.jsonl", "aggregate_metrics.csv", "candidate_decisions.jsonl"],
    }
    (results_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return processed_dir, results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="M4-v3 development-only evaluation")
    parser.add_argument("--config", default="configs/day3_dev_m4v3.yaml")
    args = parser.parse_args()
    command = subprocess.list2cmdline([sys.executable, "-m", "query_adaptive_rag.evaluation.day3_dev_m4v3", *sys.argv[1:]])
    processed, results = run(args.config, command)
    print(f"M4-v3 development run complete: {processed}; {results}")


if __name__ == "__main__":
    main()
