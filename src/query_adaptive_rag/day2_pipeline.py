"""Day 2 retrieval and query-adaptive expansion smoke pipeline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import yaml

from .chunking.text_utils import OffsetTokenizer
from .classification.complexity import classify_query
from .config import set_random_seed
from .expansion.context_expander import ExpansionRule, TokenBudgetError, assemble_context, expand_context
from .retrieval.dense import DenseRetriever


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolved_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _dependency_versions() -> dict[str, str | None]:
    packages = ("numpy", "pyarrow", "pandas", "pytz", "scikit-learn", "PyYAML",
                "datasets", "sentence-transformers", "transformers", "torch", "faiss-cpu")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _write_logs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    jsonl_path = output_dir / "per_query.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / "per_query.csv"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def run(config_path: str, query_limit: int | None = None, executed_command: str | None = None) -> tuple[Path, Path]:
    config_file = Path(config_path).resolve()
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    seed = int(raw.get("seed", 42))
    set_random_seed(seed)

    model_name = str(raw.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    tokenizer_name = str(raw.get("tokenizer_name", model_name))
    device = str(raw.get("device", "cpu"))
    top_k = int(raw.get("top_k", 5))
    token_budget = int(raw.get("token_budget", 2048))
    minimum_similarity = float(raw.get("minimum_neighbor_similarity", 0.20))
    processed_dir = _resolved_path(root, raw.get("processed_output_dir", "data/processed/day2"))
    results_dir = _resolved_path(root, raw.get("results_output_dir", "results/day2"))
    day1_dir = _resolved_path(root, raw.get("day1_input_dir", "data/processed/day1"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = day1_dir / "semantic_chunks.jsonl"
    records_path = day1_dir / "normalized.jsonl"
    if not chunk_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(f"Expected Day 1 semantic chunks and normalized records under {day1_dir}")
    chunks = _read_jsonl(chunk_path)
    records = _read_jsonl(records_path)
    if query_limit is not None:
        if query_limit < 1:
            raise ValueError("query_limit must be positive when supplied")
        records = records[:max(0, query_limit)]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for token-budget enforcement")
    encoder = SentenceTransformer(model_name, device=device)
    retriever = DenseRetriever(encoder, model_name)
    retriever.build(chunks)
    retriever.save(processed_dir / "semantic.faiss")
    with (processed_dir / "index_chunks.jsonl").open("w", encoding="utf-8") as handle:
        for row, chunk in enumerate(retriever.chunks):
            handle.write(json.dumps({
                "index_row": row,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "start": chunk["start"],
                "end": chunk["end"],
            }, ensure_ascii=False) + "\n")

    raw_rules = raw.get("expansion_rules", {})
    rules = {
        complexity: ExpansionRule(int(settings["max_depth"]), int(settings["max_expansions"]))
        for complexity, settings in raw_rules.items()
    }
    tokenizer_for_count: OffsetTokenizer = tokenizer
    result_rows = []

    for record in records:
        query = record["question"]
        classification = classify_query(query)  # Deliberately receives only the query string.
        search = retriever.retrieve(query, top_k=top_k)
        seed_chunks = [hit.chunk for hit in search.hits]
        seed_ids = [chunk["chunk_id"] for chunk in seed_chunks]
        baseline_text, baseline_tokens = assemble_context(seed_chunks, tokenizer_for_count)
        if baseline_tokens > token_budget:
            raise TokenBudgetError(
                f"query {record['id']} seed context requires {baseline_tokens} tokens, "
                f"exceeding configured budget {token_budget}"
            )

        expansion = expand_context(
            search=search,
            complexity=classification["complexity"],
            chunks=chunks,
            tokenizer=tokenizer_for_count,
            score_chunk=retriever.score,
            rules=rules,
            token_budget=token_budget,
            min_similarity=minimum_similarity,
        )
        if expansion.seed_chunk_ids != seed_ids:
            raise RuntimeError("semantic baseline and proposed method received different seed chunks")

        common = {
            "query_id": record["id"],
            "query_text": query,
            "predicted_complexity": classification["complexity"],
            "classification_features": classification["features"],
            "seed_chunk_ids": seed_ids,
            "similarity_scores": {hit.chunk["chunk_id"]: hit.score for hit in search.hits},
            "token_budget": token_budget,
            "retrieval_latency_ms": search.latency_ms,
        }
        result_rows.append({
            **common,
            "method": "semantic_baseline",
            "expanded_chunk_ids": [],
            "final_context_text": baseline_text,
            "final_context_token_count": baseline_tokens,
            "expansion_latency_ms": 0.0,
            "budget_rejected_chunk_ids": [],
            "deduplicated_chunk_ids": [],
        })
        result_rows.append({
            **common,
            "method": "adaptive_expansion",
            "expanded_chunk_ids": expansion.expanded_chunk_ids,
            "similarity_scores": expansion.similarity_scores,
            "final_context_text": expansion.context_text,
            "final_context_token_count": expansion.context_token_count,
            "expansion_latency_ms": expansion.latency_ms,
            "budget_rejected_chunk_ids": expansion.budget_rejected_chunk_ids,
            "deduplicated_chunk_ids": expansion.deduplicated_chunk_ids,
        })

    _write_logs(result_rows, results_dir)
    command = executed_command or "programmatic call to query_adaptive_rag.day2_pipeline.run"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed_command": command,
        "working_directory": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "random_seed": seed,
        "cli_overrides": {"query_limit": query_limit},
        "effective_config": {
            **raw,
            "day1_input_dir": str(day1_dir),
            "processed_output_dir": str(processed_dir),
            "results_output_dir": str(results_dir),
            "query_limit": query_limit,
        },
        "model_identifier": model_name,
        "tokenizer_identifier": tokenizer_name,
        "dependency_versions": _dependency_versions(),
        "input_counts": {"queries": len(records), "semantic_chunks": len(chunks)},
        "output_counts": {
            "result_rows": len(result_rows),
            "semantic_baseline_rows": sum(row["method"] == "semantic_baseline" for row in result_rows),
            "adaptive_expansion_rows": sum(row["method"] == "adaptive_expansion" for row in result_rows),
        },
    }
    (processed_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return processed_dir, results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 2 semantic retrieval and adaptive expansion smoke pipeline")
    parser.add_argument("--config", default="configs/day2_smoke.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit queries for a smaller smoke run")
    args = parser.parse_args()
    command = subprocess.list2cmdline([
        sys.executable, "-m", "query_adaptive_rag.day2_pipeline", *sys.argv[1:],
    ])
    processed_dir, results_dir = run(args.config, args.limit, command)
    print(f"Day 2 pipeline complete. Index and manifest: {processed_dir}; per-query results: {results_dir}")


if __name__ == "__main__":
    main()
