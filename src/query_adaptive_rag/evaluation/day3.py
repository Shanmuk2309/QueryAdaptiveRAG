"""Four-method, evidence-grounded Day 3 comparison pipeline."""

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

from ..chunking.fixed_size import fixed_size_chunks
from ..chunking.semantic import semantic_chunks
from ..classification.complexity import classify_query
from ..chunking.text_utils import OffsetTokenizer
from ..config import set_random_seed
from ..data.loaders import load_hotpotqa, load_squad
from ..evaluation.context import context_source_chunks
from ..evaluation.evidence import deduplicate_evidence
from ..evaluation.metrics import calculate_metrics
from ..evaluation.protocol import METHODS, method_protocol
from ..evaluation.splits import deterministic_split
from ..evaluation.statistics import PAIRED_METRICS, paired_bootstrap_comparisons
from ..evaluation.token_limits import audit_chunk_embedding_lengths, encoder_content_token_limit
from ..expansion.context_expander import ExpansionRule, TokenBudgetError, assemble_context, expand_context
from ..reporting.plots import write_comparison_plots
from ..retrieval.dense import DenseRetriever


DATASET_COMPLEXITY = {"squad": "single_fact", "hotpotqa": "multi_hop"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dependency_versions() -> dict[str, str | None]:
    packages = ("numpy", "pyarrow", "pandas", "pytz", "scikit-learn", "PyYAML",
                "datasets", "sentence-transformers", "transformers", "torch", "faiss-cpu", "pytest")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _validate_query_record(record: dict[str, Any]) -> None:
    required = ("id", "dataset", "question", "documents", "gold_evidence")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"query record {record.get('id', '<unknown>')} is missing {missing}")
    if record["dataset"] not in DATASET_COMPLEXITY:
        raise ValueError(f"unsupported dataset label for query {record['id']}: {record['dataset']}")
    if not record["question"].strip():
        raise ValueError(f"query {record['id']} has an empty question")
    if not record["gold_evidence"]:
        raise ValueError(f"query {record['id']} has no gold evidence; refusing to skip it")
    document_text = {document["doc_id"]: document["text"] for document in record["documents"]}
    for span in deduplicate_evidence(record["gold_evidence"]):
        text = document_text.get(span["doc_id"])
        if text is None or not (0 <= span["start"] < span["end"] <= len(text)):
            raise ValueError(f"query {record['id']} has an invalid gold evidence span: {span}")


def _unique_corpus(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for record in records:
        for document in record["documents"]:
            existing = documents.get(document["doc_id"])
            if existing is not None and existing["text"] != document["text"]:
                raise ValueError(f"document ID collision with different text: {document['doc_id']}")
            documents.setdefault(document["doc_id"], document)
    return list(documents.values())


def _budget_context(ranked_chunks: list[dict[str, Any]], tokenizer: OffsetTokenizer, budget: int):
    """Select ranked context chunks under a shared budget without truncation."""
    selected: list[dict[str, Any]] = []
    excluded: list[str] = []
    for chunk in ranked_chunks:
        proposed_text, proposed_count = assemble_context(selected + [chunk], tokenizer)
        if proposed_count <= budget:
            selected.append(chunk)
        else:
            excluded.append(chunk["chunk_id"])
    context, token_count = assemble_context(selected, tokenizer)
    if token_count > budget:
        raise TokenBudgetError(f"assembled baseline context {token_count} exceeds budget {budget}")
    return selected, context, token_count, excluded


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (*PAIRED_METRICS[:5], "context_token_count", "retrieval_latency_ms", "expansion_latency_ms")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["dataset"])].append(row)
        groups[(row["method"], "all")].append(row)
    output = []
    for (method, dataset), group_rows in sorted(groups.items()):
        row = {"method": method, "dataset": dataset, "evaluation_complexity": "mixed" if dataset == "all" else DATASET_COMPLEXITY[dataset], "query_count": len(group_rows)}
        for metric in metrics:
            row[f"mean_{metric}"] = float(np.mean([float(item[metric]) for item in group_rows]))
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty results file: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _heldout_records(raw: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    datasets = raw["datasets"]
    pool_size = int(raw["candidate_pool_size_per_dataset"])
    dev_size = int(raw["development_size_per_dataset"])
    eval_size = int(raw["evaluation_size_per_dataset"])
    output, split_manifest = [], {}
    for offset, dataset_key in enumerate(("squad", "hotpotqa")):
        settings = datasets[dataset_key]
        dataset_seed = seed + offset
        if dataset_key == "squad":
            candidate_pool = load_squad(settings.get("split", "validation"), pool_size, dataset_seed)
        else:
            candidate_pool = load_hotpotqa(
                settings.get("dataset_config", "distractor"), settings.get("split", "validation"),
                pool_size, dataset_seed,
            )
        if len(candidate_pool) < dev_size + eval_size:
            raise ValueError(
                f"{dataset_key} candidate pool returned {len(candidate_pool)} rows, "
                f"but {dev_size + eval_size} are required"
            )
        dev_records, eval_records = deterministic_split(
            candidate_pool, dev_size, eval_size, seed + 100 + offset,
        )
        output.extend(eval_records)
        split_manifest[dataset_key] = {
            "dataset_name": settings.get("dataset_name"),
            "dataset_config": settings.get("dataset_config"),
            "source_split": settings.get("split", "validation"),
            "candidate_pool_size": len(candidate_pool),
            "development_size": len(dev_records),
            "evaluation_size": len(eval_records),
            "development_ids": [record["id"] for record in dev_records],
            "evaluation_ids": [record["id"] for record in eval_records],
            "load_seed": dataset_seed,
            "split_seed": seed + 100 + offset,
        }
    return output, split_manifest


def _save_chunk_inventory(chunks: list[dict[str, Any]], path: Path) -> None:
    _write_jsonl(path, chunks)


def run(config_path: str, smoke: bool = False, executed_command: str | None = None) -> tuple[Path, Path]:
    config_file = Path(config_path).resolve()
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    seed = int(raw.get("seed", 42))
    set_random_seed(seed)

    processed_root = Path(raw.get("processed_output_dir", "data/processed/day3"))
    results_root = Path(raw.get("results_output_dir", "results/day3"))
    if not processed_root.is_absolute():
        processed_root = (root / processed_root).resolve()
    if not results_root.is_absolute():
        results_root = (root / results_root).resolve()
    processed_dir = processed_root / "smoke" if smoke else processed_root
    results_dir = results_root / "smoke" if smoke else results_root
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        day1_dir = Path(raw.get("day1_smoke_input_dir", "data/processed/day1"))
        if not day1_dir.is_absolute():
            day1_dir = (root / day1_dir).resolve()
        evaluation_records = _read_jsonl(day1_dir / "normalized.jsonl")
        split_manifest = {
            "mode": "Day 1 smoke corpus",
            "evaluation_size": len(evaluation_records),
            "evaluation_ids": [record["id"] for record in evaluation_records],
            "development_ids": [],
        }
    else:
        evaluation_records, split_manifest = _heldout_records(raw, seed)
    if not evaluation_records:
        raise ValueError("evaluation query set is empty")
    query_ids = [(record["dataset"], record["id"]) for record in evaluation_records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query IDs found in the evaluation set")
    for record in evaluation_records:
        _validate_query_record(record)
    corpus = _unique_corpus(evaluation_records)
    if not corpus:
        raise ValueError("evaluation corpus is empty")

    tokenizer_name = str(raw.get("tokenizer_name", raw["model_name"]))
    model_name = str(raw["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("fast tokenizer required for reproducible token counts")
    encoder = SentenceTransformer(model_name, device=str(raw.get("device", "cpu")))
    embedding_max_sequence_length = int(raw.get("embedding_max_sequence_length", 512))
    model_config = encoder._first_module().auto_model.config
    model_position_limit = getattr(model_config, "max_position_embeddings", embedding_max_sequence_length)
    if embedding_max_sequence_length > model_position_limit:
        raise ValueError(
            f"configured embedding_max_sequence_length={embedding_max_sequence_length} exceeds "
            f"model max_position_embeddings={model_position_limit}"
        )
    encoder.max_seq_length = embedding_max_sequence_length
    encoder_safe_m2 = bool(raw.get("encoder_safe_m2", False))
    if encoder_safe_m2:
        m2_content_tokens, tokenizer_special_token_count = encoder_content_token_limit(
            tokenizer, embedding_max_sequence_length,
        )
    else:
        # Preserve the original Day 3 protocol: M2 used 512 content tokens.
        m2_content_tokens = 512
        tokenizer_special_token_count = None
    cache_dir = Path(raw.get("embedding_cache_dir", "data/cache/sentence_embeddings"))
    if not cache_dir.is_absolute():
        cache_dir = (root / cache_dir).resolve()
    k = int(raw.get("top_k", 5))
    budget = int(raw.get("token_budget", 3072))
    overlap = int(raw.get("fixed_overlap_tokens", 32))
    rules = {
        name: ExpansionRule(int(values["max_depth"]), int(values["max_expansions"]))
        for name, values in raw["expansion_rules"].items()
    }

    methods: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS[:3]}
    chunking_times: dict[str, float] = {}
    semantic_reuse_root = raw.get("reuse_semantic_chunks_from")
    if semantic_reuse_root:
        semantic_reuse_root = Path(semantic_reuse_root)
        if not semantic_reuse_root.is_absolute():
            semantic_reuse_root = (root / semantic_reuse_root).resolve()
        if smoke:
            semantic_reuse_root = semantic_reuse_root / "smoke"
    for method in METHODS[:3]:
        started = perf_counter()
        if method == "M3_semantic" and semantic_reuse_root:
            semantic_inventory = semantic_reuse_root / "chunks" / "M3_semantic.jsonl"
            methods[method] = _read_jsonl(semantic_inventory)
            expected_doc_ids = {document["doc_id"] for document in corpus}
            actual_doc_ids = {chunk["doc_id"] for chunk in methods[method]}
            if actual_doc_ids != expected_doc_ids:
                raise ValueError(
                    "reused semantic inventory does not match this run's corpus: "
                    f"missing={len(expected_doc_ids - actual_doc_ids)}, "
                    f"unexpected={len(actual_doc_ids - expected_doc_ids)}"
                )
        else:
            for document in corpus:
                if method == "M1_fixed_256":
                    methods[method].extend(fixed_size_chunks(document["text"], document["doc_id"], tokenizer, 256, overlap))
                elif method == "M2_fixed_512":
                    methods[method].extend(fixed_size_chunks(
                        document["text"], document["doc_id"], tokenizer, m2_content_tokens, overlap,
                    ))
                else:
                    methods[method].extend(semantic_chunks(
                        document["text"], document["doc_id"], tokenizer, encoder,
                        float(raw.get("semantic_similarity_threshold", 0.45)),
                        int(raw.get("semantic_min_tokens", 80)), int(raw.get("semantic_max_tokens", 400)),
                        cache_dir, model_name,
                    ))
        chunking_times[method] = (perf_counter() - started) * 1000.0
        if not methods[method]:
            raise ValueError(f"chunking method {method} produced no chunks")
        _save_chunk_inventory(methods[method], processed_dir / "chunks" / f"{method}.jsonl")

    embedding_input_audit = None
    if encoder_safe_m2:
        # Validate the complete sequence including special tokens before passing
        # any chunk to SentenceTransformer, preventing silent model truncation.
        embedding_input_audit = {
            method: audit_chunk_embedding_lengths(
                methods[method], tokenizer, embedding_max_sequence_length,
            )
            for method in METHODS[:3]
        }

    retrievers: dict[str, DenseRetriever] = {}
    index_build_times: dict[str, float] = {}
    for method in METHODS[:3]:
        retriever = DenseRetriever(encoder, model_name)
        started = perf_counter()
        retriever.build(methods[method])
        index_build_times[method] = (perf_counter() - started) * 1000.0
        index_dir = processed_dir / "indexes"
        retriever.save(index_dir / f"{method}.faiss")
        _write_jsonl(index_dir / f"{method}_rows.jsonl", [
            {"index_row": index, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]}
            for index, chunk in enumerate(retriever.chunks)
        ])
        retrievers[method] = retriever

    chunks_by_method = {
        method: {chunk["chunk_id"]: chunk for chunk in methods[method]}
        for method in METHODS[:3]
    }
    result_rows = []
    classifier_predictions = []
    for record in evaluation_records:
        query = record["question"]
        classification = classify_query(query)  # Only query text reaches runtime classifier.
        classifier_predictions.append(classification["complexity"])
        query_searches = {
            method: retrievers[method].retrieve(query, top_k=k)
            for method in METHODS[:3]
        }
        m3_search = query_searches["M3_semantic"]
        m3_expansion = expand_context(
            search=m3_search,
            complexity=classification["complexity"],
            chunks=methods["M3_semantic"],
            tokenizer=tokenizer,
            score_chunk=retrievers["M3_semantic"].score,
            rules=rules,
            token_budget=budget,
            min_similarity=float(raw.get("minimum_neighbor_similarity", 0.20)),
        )
        if m3_expansion.seed_chunk_ids != [hit.chunk["chunk_id"] for hit in m3_search.hits]:
            raise RuntimeError("M4 seed IDs differ from M3 top-k retrieval")

        for method in METHODS[:3]:
            search = query_searches[method]
            ranked_chunks = [hit.chunk for hit in search.hits]
            context_chunks, context_text, context_count, budget_excluded = _budget_context(
                ranked_chunks, tokenizer, budget,
            )
            metric_values = calculate_metrics(ranked_chunks, context_source_chunks(context_chunks), record["gold_evidence"])
            result_rows.append({
                "query_id": record["id"],
                "dataset": record["dataset"],
                "evaluation_complexity": DATASET_COMPLEXITY[record["dataset"]],
                "predicted_complexity": classification["complexity"],
                "classification_features": classification["features"],
                "method": method,
                "query_text": query,
                "seed_chunk_ids": [hit.chunk["chunk_id"] for hit in search.hits],
                "expanded_chunk_ids": [],
                "ranked_chunk_ids": [chunk["chunk_id"] for chunk in ranked_chunks],
                "context_chunk_ids": [chunk["chunk_id"] for chunk in context_chunks],
                "similarity_scores": {hit.chunk["chunk_id"]: hit.score for hit in search.hits},
                "final_context_text": context_text,
                "evidence_span_count": len(deduplicate_evidence(record["gold_evidence"])),
                **metric_values,
                "context_token_count": context_count,
                "token_budget": budget,
                "retrieval_latency_ms": search.latency_ms,
                "expansion_latency_ms": 0.0,
                "budget_excluded_chunk_ids": budget_excluded,
            })

        m4_chunks_by_id = chunks_by_method["M3_semantic"]
        # Retrieval metrics must use the same ranked semantic seeds as M3.
        # Expansion is evaluated through final-context precision and token cost.
        m4_ranked_chunks = [hit.chunk for hit in m3_search.hits]
        m4_context_ids = m3_expansion.seed_chunk_ids + m3_expansion.expanded_chunk_ids
        m4_context_chunks = [m4_chunks_by_id[chunk_id] for chunk_id in m4_context_ids]
        metric_values = calculate_metrics(m4_ranked_chunks, context_source_chunks(m4_context_chunks), record["gold_evidence"])
        result_rows.append({
            "query_id": record["id"],
            "dataset": record["dataset"],
            "evaluation_complexity": DATASET_COMPLEXITY[record["dataset"]],
            "predicted_complexity": classification["complexity"],
            "classification_features": classification["features"],
            "method": "M4_adaptive",
            "query_text": query,
            "seed_chunk_ids": m3_expansion.seed_chunk_ids,
            "expanded_chunk_ids": m3_expansion.expanded_chunk_ids,
            "ranked_chunk_ids": [chunk["chunk_id"] for chunk in m4_ranked_chunks],
            "context_chunk_ids": m4_context_ids,
            "similarity_scores": m3_expansion.similarity_scores,
            "final_context_text": m3_expansion.context_text,
            "evidence_span_count": len(deduplicate_evidence(record["gold_evidence"])),
            **metric_values,
            "context_token_count": m3_expansion.context_token_count,
            "token_budget": budget,
            "retrieval_latency_ms": m3_search.latency_ms,
            "expansion_latency_ms": m3_expansion.latency_ms,
            "budget_excluded_chunk_ids": m3_expansion.budget_rejected_chunk_ids,
        })

    if len(result_rows) != len(evaluation_records) * len(METHODS):
        raise RuntimeError("one or more query/method result rows are missing")
    per_query_path = results_dir / "per_query.jsonl"
    _write_jsonl(per_query_path, result_rows)
    per_query_csv = results_dir / "per_query.csv"
    csv_rows = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                 for key, value in row.items()} for row in result_rows]
    fields = list(csv_rows[0])
    with per_query_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    aggregate_rows = _aggregate(result_rows)
    _write_csv(results_dir / "aggregate_metrics.csv", aggregate_rows)
    paired_rows = paired_bootstrap_comparisons(
        result_rows, repeats=int(raw.get("bootstrap_repeats", 1000)), seed=seed,
    )
    _write_csv(results_dir / "paired_m3_m4_bootstrap.csv", paired_rows)
    plots = write_comparison_plots(
        aggregate_rows, results_dir / "plots",
        m2_label=f"Fixed {m2_content_tokens} content" if encoder_safe_m2 else "Fixed 512",
    )

    command = executed_command or "programmatic call to query_adaptive_rag.evaluation.day3.run"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed_command": command,
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "random_seed": seed,
        "smoke_run": smoke,
        "model_identifier": model_name,
        "tokenizer_identifier": tokenizer_name,
        "dependency_versions": _dependency_versions(),
        "effective_config": raw,
        "method_protocol": method_protocol(
            model_name, k, budget, overlap, embedding_max_sequence_length,
            m2_content_tokens, tokenizer_special_token_count,
        ),
        "dataset_splits": split_manifest,
        "evaluation_query_count": len(evaluation_records),
        "unique_corpus_document_count": len(corpus),
        "method_builds": {
            method: {
                "chunk_count": len(methods[method]),
                "chunk_generation_latency_ms": chunking_times[method],
                "embedding_and_index_build_latency_ms": index_build_times[method],
            }
            for method in METHODS[:3]
        },
        "m4_index_reused": "M3_semantic",
        "result_rows": len(result_rows),
        "aggregate_rows": len(aggregate_rows),
        "paired_bootstrap_rows": len(paired_rows),
        "plots": [str(path.relative_to(results_dir)) for path in plots],
        "runtime_classifier_predictions": dict(
            (label, classifier_predictions.count(label))
            for label in sorted(set(classifier_predictions))
        ),
        "metrics_are_evaluation_only": True,
        **({
            "protocol_version": raw.get("protocol_version", "encoder_safe_m2_v2"),
            "embedding_input_audit": embedding_input_audit,
            "preserved_original_m2_protocol": "512 content tokens in results/day3 (potential truncation)",
            "semantic_chunks_reused_from": str(semantic_reuse_root) if semantic_reuse_root else None,
        } if encoder_safe_m2 else {}),
    }
    (results_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return processed_dir, results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 3 four-method retrieval evaluation")
    parser.add_argument("--config", default="configs/day3.yaml")
    parser.add_argument("--smoke", action="store_true", help="Evaluate only the existing Day 1 smoke corpus")
    args = parser.parse_args()
    command = subprocess.list2cmdline([
        sys.executable, "-m", "query_adaptive_rag.evaluation.day3", *sys.argv[1:],
    ])
    processed, results = run(args.config, args.smoke, command)
    print(f"Day 3 evaluation complete. Artifacts: {processed}; results: {results}")


if __name__ == "__main__":
    main()
