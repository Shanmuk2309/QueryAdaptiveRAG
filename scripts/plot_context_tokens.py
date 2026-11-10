#!/usr/bin/env python3
"""Create a paper-ready mean-context-token plot from the 10k aggregate CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "day3_heldout_10000_m4v3" / "aggregate_metrics.csv"
OUTPUT_DIR = ROOT / "paper" / "figures" / "10000_query"

# Keep the same method-to-color mapping as the recall and precision figures.
METHODS = [
    ("M1_fixed_256", "M1: Fixed 256", "#0072B2"),
    ("M2_fixed_510", "M2: Fixed 510", "#D55E00"),
    ("M3_semantic", "M3: Semantic", "#009E73"),
    ("M4_v3_fact_alias_fix", "M4-v3: Fact-level", "#CC79A7"),
]
METRIC_COLUMN = "mean_context_token_count"


def read_values() -> tuple[list[str], list[str], list[str], list[float], int]:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Aggregate metrics CSV not found: {CSV_PATH}")

    required = {"method", "dataset", "query_count", METRIC_COLUMN}
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Aggregate CSV missing required columns: {sorted(missing)}")
        rows = {row["method"]: row for row in reader if row["dataset"] == "all"}

    ids, labels, colors, values = [], [], [], []
    query_count = None
    for method_id, label, color in METHODS:
        if method_id not in rows:
            raise ValueError(f"Method {method_id} is absent from overall aggregate rows")
        row = rows[method_id]
        current_count = int(row["query_count"])
        if current_count != 10_000:
            raise ValueError(f"Expected the 10,000-query run; {method_id} has {current_count} queries")
        if query_count is not None and query_count != current_count:
            raise ValueError("Primary methods have inconsistent query counts")
        query_count = current_count
        ids.append(method_id)
        labels.append(label)
        colors.append(color)
        values.append(float(row[METRIC_COLUMN]))
    return ids, labels, colors, values, query_count or 0


def main() -> None:
    _, labels, colors, values, query_count = read_values()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    positions = list(range(len(labels)))
    bars = ax.bar(
        positions,
        values,
        width=0.66,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=9)

    ax.set_title("Mean final-context size across primary methods", pad=12)
    ax.text(
        0.5,
        1.01,
        f"{query_count:,} held-out queries (5,000 SQuAD; 5,000 HotpotQA)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#444444",
        fontsize=9,
    )
    ax.set_ylabel("Mean Context Tokens")
    ax.set_xlabel("Retrieval and context-selection method")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, 700.0)
    ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700])
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#667085")
    ax.spines["bottom"].set_color("#667085")
    ax.tick_params(colors="#333333", length=3)

    fig.text(
        0.5,
        -0.025,
        "A smaller context does not automatically imply better evidence coverage; interpret alongside Recall@k.",
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
        wrap=True,
    )

    for extension in ("png", "pdf", "svg"):
        path = OUTPUT_DIR / f"context_tokens.{extension}"
        fig.savefig(path, dpi=600 if extension == "png" else None, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
