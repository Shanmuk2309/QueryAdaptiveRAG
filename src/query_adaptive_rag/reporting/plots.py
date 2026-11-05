"""Small dependency-free SVG plots for Day 3 method comparisons."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


_COLORS = ("#3569a8", "#54a24b", "#f28e2b", "#b279a2")


def write_comparison_plots(
    aggregate_rows: list[dict[str, Any]], output_dir: str | Path, m2_label: str = "Fixed 512",
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ["M1_fixed_256", "M2_fixed_512", "M3_semantic", "M4_adaptive"]
    labels = ["Fixed 256", m2_label, "Semantic", "Adaptive"]
    overall = {row["method"]: row for row in aggregate_rows if row["dataset"] == "all"}
    written = []
    for filename, title, metrics in (
        ("retrieval_quality.svg", "Evidence retrieval quality (smoke/evaluation aggregates)",
         (("recall_at_5", "Recall@5"), ("mrr", "MRR"))),
        ("context_cost_latency.svg", "Context cost and query latency", 
         (("context_token_count", "Context tokens"), ("retrieval_latency_ms", "Retrieval latency (ms)"))),
    ):
        width, height = 900, 430
        panel_width = 400
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
        parts.append('<rect width="100%" height="100%" fill="white"/>')
        parts.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-family="Arial">{escape(title)}</text>')
        for panel, (metric, metric_label) in enumerate(metrics):
            x0 = 60 + panel * 430
            y0, chart_h = 80, 260
            vals = [float(overall.get(method, {}).get(metric, 0.0)) for method in methods]
            maximum = max(vals, default=0.0) or 1.0
            parts.append(f'<text x="{x0 + panel_width/2}" y="60" text-anchor="middle" font-size="14" font-family="Arial">{escape(metric_label)}</text>')
            parts.append(f'<line x1="{x0}" y1="{y0+chart_h}" x2="{x0+panel_width}" y2="{y0+chart_h}" stroke="#555"/>')
            slot = panel_width / len(methods)
            for index, (label, value) in enumerate(zip(labels, vals)):
                bar_h = chart_h * value / maximum
                x = x0 + index * slot + slot * 0.2
                y = y0 + chart_h - bar_h
                bar_w = slot * 0.6
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{_COLORS[index]}"/>')
                parts.append(f'<text x="{x+bar_w/2:.1f}" y="{max(y-6,y0):.1f}" text-anchor="middle" font-size="10" font-family="Arial">{value:.3f}</text>')
                parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y0+chart_h+20}" text-anchor="middle" font-size="10" font-family="Arial">{escape(label)}</text>')
        parts.append('</svg>')
        path = output_dir / filename
        path.write_text("\n".join(parts), encoding="utf-8")
        written.append(path)
    return written
