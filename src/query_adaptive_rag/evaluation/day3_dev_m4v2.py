"""Development-only comparison of M3, existing M4, and experimental M4-v2."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
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
from ..classification.complexity import classify_query
from ..chunking.text_utils import OffsetTokenizer
from ..config import set_random_seed
from ..data.loaders import _hotpot_records, _squad_records
from .context import context_source_chunks
from .day3 import _unique_corpus, _validate_query_record
from .evidence import deduplicate_evidence
from .metrics import calculate_metrics
from .splits import deterministic_split
from ..expansion.context_expander import ExpansionRule, assemble_context, expand_context
from ..expansion.context_expander_v2 import (
    DEFAULT_REDUNDANCY_THRESHOLD,
    chunk_embedding_map_from_retriever,
    expand_context_v2,
)
from ..retrieval.dense import DenseRetriever, SearchResponse


METHODS = ("M3_no_expansion", "M4_existing", "M4_v2_novelty")
DATASET_COMPLEXITY = {"squad": "single_fact", "hotpotqa": "multi_hop"}
METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "context_precision")


def _dataset_id(row: dict[str, Any], dataset_key: str) -> str:
    raw_id = row.get("id") if dataset_key == "squad" else row.get("_id", row.get("id"))
    if not isinstance(raw_id, str) or not raw_id:
        raise ValueError(f"{dataset_key} row has no usable example ID")
    return raw_id


def _development_and_heldout_ids(
    candidate_ids: list[str], development_size: int, heldout_size: int, split_seed: int,
) -> tuple[list[str], list[str]]:
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate pool contains duplicate IDs")
    dev, heldout = deterministic_split(
        [{"id": item} for item in candidate_ids], development_size, heldout_size, split_seed,
    )
    return [row["id"] for row in dev], [row["id"] for row in heldout]


def _normalize_dev_rows(rows: list[dict[str, Any]], dataset_key: str) -> list[dict[str, Any]]:
    """Normalize only already-selected development rows; held-out rows stay raw/unread."""
    if dataset_key == "squad":
        return _squad_records(rows)
    if dataset_key == "hotpotqa":
        return _hotpot_records(rows)
    raise ValueError(f"unsupported dataset key: {dataset_key}")


def _select_development_rows(
    sampled_rows: Any, dataset_key: str, development_ids: list[str],
) -> list[dict[str, Any]]:
    """Return raw rows for development IDs only, preserving split order."""
    row_map = {_dataset_id(dict(row), dataset_key): dict(row) for row in sampled_rows}
    if len(row_map) != len(sampled_rows):
        raise ValueError(f"{dataset_key} sampled rows contain duplicate IDs")
    missing = [example_id for example_id in development_ids if example_id not in row_map]
    if missing:
        raise ValueError(f"development IDs missing from {dataset_key} candidate pool: {missing[:5]}")
    return [row_map[example_id] for example_id in development_ids]


def _load_development_records(
    config: dict[str, Any], reference_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    seed = int(config["seed"])
    pool_size = int(config["candidate_pool_size_per_dataset"])
    dev_size = int(config["development_size_per_dataset"])
    heldout_size = int(config["heldout_size_per_dataset"])
    records: list[dict[str, Any]] = []
    split_info: dict[str, Any] = {}

    for offset, dataset_key in enumerate(("squad", "hotpotqa")):
        settings = config["datasets"][dataset_key]
        load_seed = seed + offset
        split_seed = seed + 100 + offset
        dataset_name = settings["dataset_name"]
        dataset_config = settings.get("dataset_config")
        split = settings.get("split", "validation")
        if dataset_config:
            raw_rows = load_dataset(dataset_name, dataset_config, split=split)
        else:
            raw_rows = load_dataset(dataset_name, split=split)
        sampled_rows = raw_rows.shuffle(seed=load_seed).select(range(min(pool_size, len(raw_rows))))
        candidate_ids = [_dataset_id(dict(row), dataset_key) for row in sampled_rows]
        if len(candidate_ids) < dev_size + heldout_size:
            raise ValueError(
                f"{dataset_key} pool has {len(candidate_ids)} rows; "
                f"{dev_size + heldout_size} are required"
            )
        dev_ids, heldout_ids = _development_and_heldout_ids(
            candidate_ids, dev_size, heldout_size, split_seed,
        )
        reference = reference_manifest.get("dataset_splits", {}).get(dataset_key)
        if reference is None:
            raise ValueError(f"reference manifest has no {dataset_key} split")
        expected_dev_ids = reference.get("development_ids", [])
        expected_heldout_ids = reference.get("evaluation_ids", [])
        if dev_ids != expected_dev_ids:
            raise ValueError(f"reconstructed {dataset_key} development IDs differ from the Day 3 manifest")
        if heldout_ids != expected_heldout_ids:
            raise ValueError(f"reconstructed {dataset_key} held-out IDs differ from the Day 3 manifest")
        if set(dev_ids) & set(heldout_ids):
            raise ValueError(f"{dataset_key} development and held-out IDs overlap")

        # Select rows by development IDs before normalization. Held-out question,
        # answer, document, and evidence payloads are never normalized or used.
        selected_raw_rows = _select_development_rows(sampled_rows, dataset_key, dev_ids)
        normalized_dev = _normalize_dev_rows(selected_raw_rows, dataset_key)
        if [row["id"] for row in normalized_dev] != dev_ids:
            raise ValueError(f"normalization changed {dataset_key} development ID order or count")
        records.extend(normalized_dev)
        split_info[dataset_key] = {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "source_split": split,
            "candidate_pool_count": len(candidate_ids),
            "development_count": len(dev_ids),
            "heldout_count": len(heldout_ids),
            "loader_seed": load_seed,
            "split_seed": split_seed,
            "development_ids": dev_ids,
            "development_matches_reference_manifest": True,
            "development_heldout_overlap_count": 0,
            "heldout_payloads_normalized_or_used": False,
        }
    return records, split_info


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dependency_versions() -> dict[str, str | None]:
    package_names = ("numpy", "datasets", "sentence-transformers", "transformers", "torch", "faiss-cpu", "PyYAML")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in package_names:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        for dataset in ("squad", "hotpotqa", "all"):
            group = [row for row in rows if row["method"] == method and (dataset == "all" or row["dataset"] == dataset)]
            if not group:
                continue
            result: dict[str, Any] = {
                "method": method,
                "dataset": dataset,
                "evaluation_complexity": "mixed" if dataset == "all" else DATASET_COMPLEXITY[dataset],
                "query_count": len(group),
            }
            for metric in (*METRICS, "context_token_count", "retrieval_latency_ms", "expansion_latency_ms"):
                result[f"mean_{metric}"] = float(np.mean([float(row[metric]) for row in group]))
            for count_key in (
                "candidate_neighbors_considered", "neighbors_accepted", "neighbors_rejected_redundant",
                "neighbors_rejected_irrelevant", "neighbors_rejected_budget", "neighbors_deduplicated",
            ):
                result[f"total_{count_key}"] = sum(int(row[count_key]) for row in group)
            result["redundancy_filter_applied"] = method == "M4_v2_novelty"
            output.append(result)
    return output


def evaluate_query_methods(
    record: dict[str, Any],
    search: SearchResponse,
    chunks: list[dict[str, Any]],
    retriever: DenseRetriever,
    chunk_embeddings: dict[str, np.ndarray],
    tokenizer: OffsetTokenizer,
    rules: dict[str, ExpansionRule],
    token_budget: int,
    min_similarity: float,
    redundancy_threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate three contexts against one already-computed M3 SearchResponse."""
    query = record["question"]
    classification = classify_query(query)
    seeds = [hit.chunk for hit in search.hits]
    seed_ids = [hit.chunk["chunk_id"] for hit in search.hits]
    seed_context, seed_token_count = assemble_context(seeds, tokenizer)
    if seed_token_count > token_budget:
        raise ValueError(f"M3 seed context for query {record['id']} exceeds the token budget")
    v1 = expand_context(
        search, classification["complexity"], chunks, tokenizer, retriever.score,
        rules, token_budget, min_similarity,
    )
    v2 = expand_context_v2(
        search, classification["complexity"], chunks, chunk_embeddings, tokenizer,
        retriever.score, rules, token_budget, min_similarity, redundancy_threshold,
    )
    for result in (v1, v2):
        if result.seed_chunk_ids != seed_ids:
            raise RuntimeError("an expansion method changed the shared M3 seed IDs")

    candidate_count_v1 = max(0, len(v1.similarity_scores) - len(seed_ids))
    candidate_count_v2 = max(0, len(v2.similarity_scores) - len(seed_ids))
    contexts = (
        ("M3_no_expansion", seeds, seed_context, seed_token_count, 0.0, [], 0, 0, 0, 0, 0, 0, False),
        (
            "M4_existing", [*seeds, *[next(chunk for chunk in chunks if chunk["chunk_id"] == cid)
                                      for cid in v1.expanded_chunk_ids]],
            v1.context_text, v1.context_token_count, v1.latency_ms, v1.expanded_chunk_ids,
            candidate_count_v1, len(v1.expanded_chunk_ids), 0,
            sum(score < min_similarity for cid, score in v1.similarity_scores.items() if cid not in seed_ids),
            len(v1.budget_rejected_chunk_ids), len(v1.deduplicated_chunk_ids), False,
        ),
        (
            "M4_v2_novelty", [*seeds, *[next(chunk for chunk in chunks if chunk["chunk_id"] == cid)
                                        for cid in v2.expanded_chunk_ids]],
            v2.context_text, v2.context_token_count, v2.latency_ms, v2.expanded_chunk_ids,
            candidate_count_v2, len(v2.expanded_chunk_ids), len(v2.redundancy_rejected_chunk_ids),
            len(v2.similarity_rejected_chunk_ids), len(v2.budget_rejected_chunk_ids),
            len(v2.deduplicated_chunk_ids), True,
        ),
    )

    output = []
    for (method, context_chunks, context_text, context_token_count, expansion_latency, expanded_ids,
         considered, accepted, redundant_rejected, irrelevant_rejected, budget_rejected,
         deduplicated, redundancy_filter_applied) in contexts:
        # Gold evidence appears only in this metric call, after classification,
        # M3 retrieval, and both context-selection methods have completed.
        metrics = calculate_metrics(seeds, context_source_chunks(context_chunks), record["gold_evidence"])
        output.append({
            "query_id": record["id"],
            "dataset": record["dataset"],
            "evaluation_complexity": DATASET_COMPLEXITY[record["dataset"]],
            "predicted_complexity": classification["complexity"],
            "classification_features": classification["features"],
            "method": method,
            "query_text": query,
            "seed_chunk_ids": seed_ids,
            "expanded_chunk_ids": expanded_ids,
            "similarity_scores": {hit.chunk["chunk_id"]: float(hit.score) for hit in search.hits},
            "neighbor_similarity_scores": {
                chunk_id: float(score)
                for chunk_id, score in (
                    v1.similarity_scores.items() if method == "M4_existing"
                    else v2.similarity_scores.items() if method == "M4_v2_novelty"
                    else []
                )
                if chunk_id not in seed_ids
            },
            "neighbor_redundancy_scores": (
                v2.redundancy_scores if method == "M4_v2_novelty" else {}
            ),
            "final_context_text": context_text,
            **metrics,
            "context_token_count": context_token_count,
            "token_budget": token_budget,
            "retrieval_latency_ms": search.latency_ms,
            "expansion_latency_ms": expansion_latency,
            "candidate_neighbors_considered": considered,
            "neighbors_accepted": accepted,
            "neighbors_rejected_redundant": redundant_rejected,
            "neighbors_rejected_irrelevant": irrelevant_rejected,
            "neighbors_rejected_budget": budget_rejected,
            "neighbors_deduplicated": deduplicated,
            "redundancy_filter_applied": redundancy_filter_applied,
        })
    return output


def _render_report(aggregate: list[dict[str, Any]], config: dict[str, Any], query_count: int) -> str:
    lines = [
        "# Development-Only M3 / M4 / M4-v2 Comparison",
        "",
        f"Evaluated {query_count} existing development queries (50 SQuAD and 50 HotpotQA).",
        "All three methods received the same M3 seed chunks and query embedding per query.",
        f"M4-v2 redundancy threshold: {config['redundancy_threshold']:.2f} (fixed; not tuned).",
        "",
        "Recall@k and MRR use the unchanged M3 seed ranking for all three methods. "
        "Context precision and token count use each method's final rendered context.",
        "",
        "| Method | Dataset | n | R@1 | R@3 | R@5 | MRR | Context precision | Context tokens | Expansion ms | Candidates | Accepted | Redundant rejects |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['method']} | {row['dataset']} | {row['query_count']} | "
            f"{row['mean_recall_at_1']:.3f} | {row['mean_recall_at_3']:.3f} | "
            f"{row['mean_recall_at_5']:.3f} | {row['mean_mrr']:.3f} | "
            f"{row['mean_context_precision']:.3f} | {row['mean_context_token_count']:.1f} | "
            f"{row['mean_expansion_latency_ms']:.2f} | {row['total_candidate_neighbors_considered']} | "
            f"{row['total_neighbors_accepted']} | {row['total_neighbors_rejected_redundant']} |"
        )
    lines.extend([
        "",
        "`Redundant rejects` is zero for M3/M4 because only M4-v2 applies a semantic redundancy filter; "
        "the aggregate CSV includes `redundancy_filter_applied` to distinguish not-applied from no candidates.",
        "Candidate neighbors considered means valid, unvisited same-document neighbors that reached query scoring.",
        "Evidence annotations were used only to calculate metrics. No held-out query payloads were normalized or evaluated.",
        "",
    ])
    return "\n".join(lines)


def run(config_path: str, executed_command: str | None = None) -> tuple[Path, Path]:
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    seed = int(config.get("seed", 42))
    set_random_seed(seed)

    processed_dir = (root / config["processed_output_dir"]).resolve()
    results_dir = (root / config["results_output_dir"]).resolve()
    protected_dirs = ((root / "data/processed/day3").resolve(), (root / "data/processed/day3_corrected").resolve(),
                      (root / "results/day3").resolve(), (root / "results/day3_corrected").resolve())
    for output_dir in (processed_dir, results_dir):
        if output_dir in protected_dirs or any(output_dir == protected or protected in output_dir.parents for protected in protected_dirs):
            raise ValueError(f"development experiment output must be isolated from existing Day 3 outputs: {output_dir}")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing development experiment output: {output_dir}")

    reference_path = (root / config["reference_manifest"]).resolve()
    reference_manifest = json.loads(reference_path.read_text(encoding="utf-8"))
    records, split_info = _load_development_records(config, reference_manifest)
    if len(records) != 100:
        raise ValueError(f"expected 100 development queries, got {len(records)}")
    if sum(record["dataset"] == "squad" for record in records) != 50:
        raise ValueError("development split must contain 50 SQuAD examples")
    if sum(record["dataset"] == "hotpotqa" for record in records) != 50:
        raise ValueError("development split must contain 50 HotpotQA examples")
    for record in records:
        _validate_query_record(record)
    corpus = _unique_corpus(records)

    tokenizer_name = str(config.get("tokenizer_name", config["model_name"]))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("fast tokenizer required for reproducible token counts")
    model_name = str(config["model_name"])
    encoder = SentenceTransformer(model_name, device=str(config.get("device", "cpu")))
    model_limit = int(config.get("embedding_max_sequence_length", 512))
    encoder_config = encoder._first_module().auto_model.config
    max_positions = int(getattr(encoder_config, "max_position_embeddings", model_limit))
    if model_limit > max_positions:
        raise ValueError(f"configured encoder limit {model_limit} exceeds model position limit {max_positions}")
    encoder.max_seq_length = model_limit

    cache_dir = (root / config.get("embedding_cache_dir", "data/cache/day3_dev_m4v2")).resolve()
    chunks: list[dict[str, Any]] = []
    chunking_started = perf_counter()
    for document in corpus:
        chunks.extend(semantic_chunks(
            document["text"], document["doc_id"], tokenizer, encoder,
            float(config.get("semantic_similarity_threshold", 0.45)),
            int(config.get("semantic_min_tokens", 80)),
            int(config.get("semantic_max_tokens", 400)), cache_dir, model_name,
        ))
    chunking_latency_ms = (perf_counter() - chunking_started) * 1000.0
    if not chunks:
        raise ValueError("semantic chunking produced no M3 chunks")

    retriever = DenseRetriever(encoder, model_name)
    index_started = perf_counter()
    retriever.build(chunks)
    index_build_latency_ms = (perf_counter() - index_started) * 1000.0
    chunk_embeddings = chunk_embedding_map_from_retriever(retriever)

    rules = {
        key: ExpansionRule(int(value["max_depth"]), int(value["max_expansions"]))
        for key, value in config["expansion_rules"].items()
    }
    top_k = int(config.get("top_k", 5))
    budget = int(config.get("token_budget", 3072))
    min_similarity = float(config.get("minimum_neighbor_similarity", 0.20))
    redundancy_threshold = float(config.get("redundancy_threshold", DEFAULT_REDUNDANCY_THRESHOLD))
    result_rows = []
    for record in records:
        # M3 is retrieved exactly once. Its SearchResponse (seeds and query vector)
        # is shared by baseline rendering, existing M4, and M4-v2.
        search = retriever.retrieve(record["question"], top_k=top_k)
        result_rows.extend(evaluate_query_methods(
            record, search, chunks, retriever, chunk_embeddings, tokenizer,
            rules, budget, min_similarity, redundancy_threshold,
        ))
    if len(result_rows) != len(records) * len(METHODS):
        raise RuntimeError("missing one or more per-query method results")

    chunk_dir = processed_dir / "chunks"
    _write_jsonl(chunk_dir / "M3_semantic.jsonl", chunks)
    retriever.save(processed_dir / "indexes" / "M3_semantic.faiss")
    _write_jsonl(processed_dir / "indexes" / "M3_semantic_rows.jsonl", [
        {"index_row": i, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]}
        for i, chunk in enumerate(retriever.chunks)
    ])

    results_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(results_dir / "per_query.jsonl", result_rows)
    csv_rows = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                 for key, value in row.items()} for row in result_rows]
    _write_csv(results_dir / "per_query.csv", csv_rows)
    aggregates = _aggregate(result_rows)
    _write_csv(results_dir / "aggregate_metrics.csv", aggregates)
    report = _render_report(aggregates, config, len(records))
    (results_dir / "development_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed_command": executed_command or "programmatic call to day3_dev_m4v2.run",
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "experiment_name": config.get("experiment_name", "day3_dev_m4v2"),
        "config": config,
        "protocol": {
            "methods": list(METHODS),
            "seed_retrieval_runs_per_query": 1,
            "same_m3_search_response_shared_across_methods": True,
            "same_corpus_embedding_model_top_k_and_token_budget": True,
            "redundancy_threshold": redundancy_threshold,
            "redundancy_threshold_tuned_automatically": False,
            "candidate_count_definition": "valid unvisited same-document neighbors reaching query scoring",
            "gold_evidence_used_for": "development metric computation only",
            "classifier_runtime_input": "query text only",
        },
        "reference_manifest": str(reference_path),
        "dataset_splits": split_info,
        "development_query_count": len(records),
        "corpus_document_count": len(corpus),
        "semantic_chunk_count": len(chunks),
        "semantic_chunk_generation_latency_ms": chunking_latency_ms,
        "embedding_and_index_build_latency_ms": index_build_latency_ms,
        "result_rows": len(result_rows),
        "aggregate_rows": len(aggregates),
        "heldout_query_payloads_normalized_or_used": False,
        "heldout_query_text_or_evidence_used_for_tuning": False,
        "outputs": [
            "per_query.csv", "per_query.jsonl", "aggregate_metrics.csv", "development_report.md",
            "../data/processed/day3_dev_m4v2/chunks/M3_semantic.jsonl",
            "../data/processed/day3_dev_m4v2/indexes/M3_semantic.faiss",
            "../data/processed/day3_dev_m4v2/indexes/M3_semantic_rows.jsonl",
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return processed_dir, results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-only M3/M4/M4-v2 comparison")
    parser.add_argument("--config", default="configs/day3_dev_m4v2.yaml")
    args = parser.parse_args()
    command = subprocess.list2cmdline([
        sys.executable, "-m", "query_adaptive_rag.evaluation.day3_dev_m4v2", *sys.argv[1:],
    ])
    processed, results = run(args.config, command)
    print(f"Development comparison complete. Artifacts: {processed}; results: {results}")


if __name__ == "__main__":
    main()
