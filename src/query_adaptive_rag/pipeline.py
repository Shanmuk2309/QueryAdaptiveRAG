"""Day 1 command-line pipeline: normalize data and build chunk sets."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .chunking.fixed_size import fixed_size_chunks
from .chunking.semantic import semantic_chunks
from .config import load_config, set_random_seed
from .data.loaders import load_configured_datasets


_DEPENDENCIES = {
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "pandas": "pandas",
    "pytz": "pytz",
    "scikit-learn": "scikit-learn",
    "PyYAML": "PyYAML",
    "datasets": "datasets",
    "sentence-transformers": "sentence-transformers",
    "transformers": "transformers",
    "torch": "torch",
    "faiss-cpu": "faiss-cpu",
    "pytest": "pytest",
}


def _dependency_versions() -> dict[str, str | None]:
    versions = {"python": platform.python_version()}
    for label, package in _DEPENDENCIES.items():
        try:
            versions[label] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def run(
    config_path: str,
    limit_override: int | None = None,
    executed_command: str | None = None,
) -> Path:
    config = load_config(config_path)
    set_random_seed(config.seed)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast Hugging Face tokenizer is required for source offsets")
    model = SentenceTransformer(config.model_name, device=config.device)

    dataset_settings = {name: dict(settings) for name, settings in config.datasets.items()}
    if limit_override is not None:
        for settings in dataset_settings.values():
            settings["limit"] = limit_override
    records = load_configured_datasets(dataset_settings, config.seed)
    normalized_path = config.output_dir / "normalized.jsonl"
    with normalized_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    methods = {f"fixed_{size}": [] for size in config.fixed_chunk_sizes}
    methods["semantic"] = []
    unique_documents = {}
    for record in records:
        for document in record["documents"]:
            unique_documents.setdefault(document["doc_id"], document)
    for document in unique_documents.values():
        for size in config.fixed_chunk_sizes:
            methods[f"fixed_{size}"].extend(fixed_size_chunks(
                document["text"], document["doc_id"], tokenizer, size, config.fixed_overlap_tokens,
            ))
        methods["semantic"].extend(semantic_chunks(
            document["text"], document["doc_id"], tokenizer, model,
            config.semantic_similarity_threshold, config.semantic_min_tokens,
            config.semantic_max_tokens, config.cache_dir / "sentence_embeddings", config.model_name,
        ))

    for method, chunks in methods.items():
        with (config.output_dir / f"{method}_chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    effective_config = asdict(config)
    effective_config["fixed_chunk_sizes"] = list(config.fixed_chunk_sizes)
    effective_config["output_dir"] = str(config.output_dir)
    effective_config["cache_dir"] = str(config.cache_dir)
    effective_config["datasets"] = dataset_settings
    dataset_counts = dict(Counter(record["dataset"] for record in records))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed_command": executed_command or "programmatic call to query_adaptive_rag.pipeline.run",
        "working_directory": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "random_seed": config.seed,
        "cli_overrides": {"limit_per_dataset": limit_override},
        "effective_config": effective_config,
        "model_identifier": config.model_name,
        "tokenizer_identifier": config.tokenizer_name,
        "dependency_versions": _dependency_versions(),
        "dataset_counts": {"total": len(records), "by_dataset": dataset_counts},
        "unique_document_count": len(unique_documents),
        "chunk_counts": {method: len(chunks) for method, chunks in methods.items()},
    }
    (config.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return config.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize SQuAD/HotpotQA and generate Day 1 chunks")
    parser.add_argument("--config", default="configs/day1.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Override examples per dataset (useful for smoke runs)")
    args = parser.parse_args()
    command = subprocess.list2cmdline([
        sys.executable, "-m", "query_adaptive_rag.pipeline", *sys.argv[1:],
    ])
    output_dir = run(args.config, args.limit, command)
    print(f"Day 1 pipeline complete. Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
