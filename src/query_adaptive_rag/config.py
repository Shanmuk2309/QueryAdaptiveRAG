"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    model_name: str
    device: str
    tokenizer_name: str
    semantic_similarity_threshold: float
    semantic_min_tokens: int
    semantic_max_tokens: int
    fixed_chunk_sizes: tuple[int, ...]
    fixed_overlap_tokens: int
    datasets: dict[str, dict[str, Any]]
    output_dir: Path
    cache_dir: Path


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML settings and resolve relative paths from the repo root."""
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (root / candidate).resolve()

    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        model_name=str(raw.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        device=str(raw.get("device", "cpu")),
        tokenizer_name=str(raw.get("tokenizer_name", raw.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))),
        semantic_similarity_threshold=float(raw.get("semantic_similarity_threshold", 0.45)),
        semantic_min_tokens=int(raw.get("semantic_min_tokens", 80)),
        semantic_max_tokens=int(raw.get("semantic_max_tokens", 400)),
        fixed_chunk_sizes=tuple(int(size) for size in raw.get("fixed_chunk_sizes", [256, 512])),
        fixed_overlap_tokens=int(raw.get("fixed_overlap_tokens", 32)),
        datasets=dict(raw.get("datasets", {})),
        output_dir=resolve(str(raw.get("output_dir", "data/processed/day1"))),
        cache_dir=resolve(str(raw.get("cache_dir", "data/cache"))),
    )


def set_random_seed(seed: int) -> None:
    """Seed available CPU-side random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
