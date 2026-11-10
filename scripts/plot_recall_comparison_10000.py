#!/usr/bin/env python3
"""Plot primary-method recall metrics from the 10,000-query held-out run."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "results" / "day3_heldout_10000_m4v3" / "aggregate_metrics.csv"
FIGURE_DIR = ROOT / "paper" / "figures" / "10000_query"

METHODS = [
    ("M1_fixed_256", "M1: Fixed 256", "#0072B2"),
    ("M2_fixed_510", "M2: Fixed 510", "#D55E00"),
    ("M3_semantic", "M3: Semantic", "#009E73"),
    ("M4_v3_fact_alias_fix", "M4-v3: Fact-level", "#CC79A7"),
]
METRICS = [
    ("mean_recall_at_1", "Recall@1"),
    ("mean_recall_at_3", "Recall@3"),
    ("mean_recall_at_5", "Recall@5"),
]


def load_overall_metrics() -> dict[str, dict[str, float]]:
    """Read the aggregate CSV's all-query rows and validate required fields."""
    if not METRICS_CSV.is_file():
        raise FileNotFoundError(f"Aggregate metrics CSV not found: {METRICS_CSV}")

    required = {"method", "dataset", "query_count", *(column for column, _ in METRICS)}
    with METRICS_CSV.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Aggregate metrics CSV is missing columns: {sorted(missing)}")
        rows = [row for row in reader if row["dataset"] == "all"]

    results: dict[str, dict[str, float]] = {}
    for row in rows:
        results[row["method"]] = {column: float(row[column]) for column, _ in METRICS}
        if int(row["query_count"]) != 10_000:
            raise ValueError(
                f"Expected 10,000 overall queries for {row['method']}; "
                f"found {row['query_count']}"
            )

    expected_methods = {method for method, _, _ in METHODS}
    if set(results) & expected_methods != expected_methods:
        absent = sorted(expected_methods.difference(results))
        raise ValueError(f"Required primary methods are absent from aggregate metrics: {absent}")
    return results


def main() -> None:
    values = load_overall_metrics()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # A 0–1 scale preserves the full possible recall range and avoids truncation.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    metric_positions = np.arange(len(METRICS), dtype=float)
    bar_width = 0.18
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * bar_width

    for method_index, (method, label, color) in enumerate(METHODS):
        scores = [values[method][column] for column, _ in METRICS]
        bars = ax.bar(
            metric_positions + offsets[method_index],
            scores,
            width=bar_width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{score:.4f}" for score in scores],
            padding=3,
            fontsize=7.5,
            rotation=90,
            label_type="edge",
        )

    ax.set_title("Retrieval recall across four primary RAG methods", pad=12)
    ax.text(
        0.5,
        1.01,
        "10,000 held-out queries (5,000 SQuAD; 5,000 HotpotQA)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    ax.set_ylabel("Mean evidence recall")
    ax.set_xlabel("Retrieval metric")
    ax.set_xticks(metric_positions, [label for _, label in METRICS])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.65, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#667085")
    ax.spines["bottom"].set_color("#667085")
    ax.tick_params(axis="both", colors="#333333", length=3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=1.4,
    )

    caption = (
        "Recall@k is the fraction of unique gold evidence spans covered by the first k ranked chunks. "
        "All bars use the full 0–1 recall scale."
    )
    fig.text(0.5, -0.015, caption, ha="center", va="top", fontsize=8, color="#444444", wrap=True)

    for extension in ("png", "pdf", "svg"):
        destination = FIGURE_DIR / f"recall_comparison.{extension}"
        fig.savefig(destination, dpi=600 if extension == "png" else None, bbox_inches="tight")
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
