"""Paired bootstrap summaries for method-level Day 3 comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


PAIRED_METRICS = (
    "recall_at_1", "recall_at_3", "recall_at_5", "mrr", "context_precision",
    "context_token_count", "retrieval_latency_ms", "expansion_latency_ms",
)


def paired_bootstrap_comparisons(
    rows: list[dict[str, Any]],
    method_a: str = "M3_semantic",
    method_b: str = "M4_adaptive",
    repeats: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Estimate mean method_b-minus-method_a deltas and percentile 95% CIs.

    Rows are paired on ``(dataset, query_id)``. Intervals are descriptive; this
    helper does not calculate p-values or declare statistical significance.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    by_query: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["method"] in {method_a, method_b}:
            by_query[(row["dataset"], row["query_id"])][row["method"]] = row

    groups: list[tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]]] = []
    for dataset in sorted({key[0] for key in by_query}):
        pairs = [
            (methods[method_a], methods[method_b])
            for (row_dataset, _), methods in by_query.items()
            if row_dataset == dataset and method_a in methods and method_b in methods
        ]
        groups.append((dataset, pairs))
    overall = [
        (methods[method_a], methods[method_b])
        for methods in by_query.values()
        if method_a in methods and method_b in methods
    ]
    groups.append(("all", overall))

    rng = np.random.default_rng(seed)
    output = []
    for group, pairs in groups:
        if not pairs:
            continue
        for metric in PAIRED_METRICS:
            differences = np.asarray([float(b[metric]) - float(a[metric]) for a, b in pairs], dtype=np.float64)
            sample_indices = rng.integers(0, len(differences), size=(repeats, len(differences)))
            sample_means = differences[sample_indices].mean(axis=1)
            lower, upper = np.quantile(sample_means, [0.025, 0.975])
            output.append({
                "comparison": f"{method_b} - {method_a}",
                "dataset": group,
                "metric": metric,
                "paired_queries": len(differences),
                "mean_delta": float(differences.mean()),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "bootstrap_repeats": repeats,
                "bootstrap_seed": seed,
            })
    return output
