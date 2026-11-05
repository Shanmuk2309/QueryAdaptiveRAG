"""Deterministic held-out development/evaluation partitioning."""

from __future__ import annotations

import random
from typing import Any


def deterministic_split(
    records: list[dict[str, Any]],
    development_size: int,
    evaluation_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return disjoint development and evaluation records from a candidate pool."""
    if development_size < 0 or evaluation_size < 1:
        raise ValueError("development_size must be non-negative and evaluation_size must be positive")
    if development_size + evaluation_size > len(records):
        raise ValueError("candidate pool is smaller than requested development/evaluation sizes")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    development = [records[index] for index in indices[:development_size]]
    evaluation = [records[index] for index in indices[development_size:development_size + evaluation_size]]
    return development, evaluation
