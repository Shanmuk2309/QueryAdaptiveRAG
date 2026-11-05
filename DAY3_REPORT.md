# Day 3 Comparative Evaluation Report

## Run summary

The held-out evaluation completed on 400 questions: 200 SQuAD validation questions (single-fact group) and 200 HotpotQA distractor validation questions (multi-hop group). For each dataset, 250 seeded candidates were loaded, then split into 50 development and 200 evaluation IDs. The development IDs were reserved and not used for tuning. The shared corpus contained 2,185 unique documents. Every query was evaluated by all four methods, producing 1,600 per-query rows. No query was skipped.

The run used seed 42, CPU, `sentence-transformers/all-MiniLM-L6-v2`, a shared 512-token encoder limit, top-k 5, a 3,072-token rendered-context budget, and 32-token overlap for both fixed baselines. Semantic chunking used threshold 0.45, 80-token minimum, and 400-token maximum. M4 reused exactly the M3 seed chunks and ranked seeds for retrieval metrics; only its final context was expanded. The effective configuration, split IDs, runtime package versions, command, chunk/index timings, and output counts are in `results/day3/experiment_manifest.json`.

## Metric definitions

- **Recall@k (k = 1, 3, 5):** number of unique gold evidence spans overlapping at least one of the first k ranked retrieval chunks, divided by the number of deduplicated gold spans. Evidence is deduplicated by document ID and half-open character offsets.
- **MRR:** reciprocal rank of the first ranked chunk overlapping any gold evidence span; zero when none is found.
- **Context Precision:** proportion of unique chunks that contribute text to the rendered context and overlap gold evidence. Exact duplicate text and fully covered overlap are excluded; partially overlapping chunks are scored only on the newly contributing source interval.
- **Context token count:** tokens in the final rendered context counted with the shared model tokenizer, excluding special tokens.
- **Retrieval latency:** query embedding plus FAISS search, excluding corpus chunking and index construction. M4 reuses M3’s search response, so retrieval latency is identical by design.
- **Expansion latency:** M4 neighbor selection and context assembly time. It is zero for M1–M3.

Gold evidence annotations are used only after retrieval, in metric computation. The runtime classifier receives question text only. SQuAD and HotpotQA are reported as their intended single-fact and multi-hop evaluation groups; classifier predictions are separately recorded in the manifest and per-query rows.

## Aggregate results

Means below are computed over query-level measurements. Retrieval scores for M4 intentionally equal M3 because both use the same semantic seed ranking; this is the controlled retrieval comparison. M4’s context precision and token count measure the expanded final context.

| Method | Dataset | n | Recall@1 | Recall@3 | Recall@5 | MRR | Context precision | Context tokens | Retrieval ms | Expansion ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 fixed 256 | All | 400 | 0.614 | 0.792 | 0.858 | 0.876 | 0.247 | 584.4 | 66.74 | 0.00 |
| M2 fixed 512 | All | 400 | 0.616 | 0.795 | 0.858 | 0.877 | 0.247 | 606.6 | 59.72 | 0.00 |
| M3 semantic | All | 400 | 0.601 | 0.773 | 0.846 | 0.865 | 0.244 | 526.3 | 60.89 | 0.00 |
| M4 adaptive | All | 400 | 0.601 | 0.773 | 0.846 | 0.865 | 0.232 | 557.4 | 60.89 | 9.46 |
| M1 fixed 256 | SQuAD | 200 | 0.820 | 0.920 | 0.950 | 0.875 | 0.190 | 675.1 | 82.06 | 0.00 |
| M2 fixed 512 | SQuAD | 200 | 0.825 | 0.925 | 0.950 | 0.878 | 0.190 | 717.1 | 72.67 | 0.00 |
| M3 semantic | SQuAD | 200 | 0.793 | 0.900 | 0.945 | 0.856 | 0.190 | 594.7 | 74.63 | 0.00 |
| M4 adaptive | SQuAD | 200 | 0.793 | 0.900 | 0.945 | 0.856 | 0.183 | 621.5 | 74.63 | 11.28 |
| M1 fixed 256 | HotpotQA | 200 | 0.407 | 0.665 | 0.765 | 0.877 | 0.304 | 493.7 | 51.41 | 0.00 |
| M2 fixed 512 | HotpotQA | 200 | 0.407 | 0.665 | 0.765 | 0.877 | 0.304 | 496.1 | 46.77 | 0.00 |
| M3 semantic | HotpotQA | 200 | 0.409 | 0.645 | 0.748 | 0.875 | 0.297 | 457.8 | 47.15 | 0.00 |
| M4 adaptive | HotpotQA | 200 | 0.409 | 0.645 | 0.748 | 0.875 | 0.281 | 493.3 | 47.15 | 7.63 |

Paired M4-minus-M3 bootstrap intervals use 1,000 resamples at seed 42. Across all 400 pairs, the mean Context Precision difference was **-0.0118** (95% percentile interval **[-0.0141, -0.0095]**); the mean context-token difference was **+31.1 tokens** (95% interval **[+24.8, +38.1]**). For HotpotQA, the corresponding differences were **-0.0164** (95% interval **[-0.0207, -0.0125]**) and **+35.5 tokens** (95% interval **[+26.6, +45.8]**). For SQuAD, they were **-0.0073** (95% interval **[-0.0095, -0.0051]**) and **+26.7 tokens** (95% interval **[+18.2, +35.5]**). Recall@k and MRR paired differences are zero by the controlled seed-ranking protocol. These descriptive bootstrap intervals are reported without a significance claim.

## Artifacts and reproduction

- Per-query records: `results/day3/per_query.csv` and `results/day3/per_query.jsonl`
- Aggregate table: `results/day3/aggregate_metrics.csv`
- Paired estimates: `results/day3/paired_m3_m4_bootstrap.csv`
- Run manifest: `results/day3/experiment_manifest.json`
- Plots: `results/day3/plots/retrieval_quality.svg` and `results/day3/plots/context_cost_latency.svg`
- Chunk inventories and FAISS indexes: `data/processed/day3/`

Commands from the repository root (PowerShell):

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m query_adaptive_rag.evaluation.day3 --config configs/day3.yaml --smoke
.\.venv\Scripts\python.exe -m query_adaptive_rag.evaluation.day3 --config configs/day3.yaml
```

## Limitations and interpretation

The subset is a seeded sample of each dataset’s validation split, not the full datasets; all selected evaluation documents form the retrieval corpus. The data and model must be available from Hugging Face for a fresh run. CPU index construction dominates execution time. The tokenizer emitted a warning for a 532-token sequence against the configured 512-token encoder limit during the full run; the pipeline completed and produced finite outputs, but this indicates a model-input truncation/length edge case that merits investigation before treating these measurements as final paper evidence. The fixed 512-token chunk size is also at the encoder boundary, where special tokens can reduce the available content window. No answer generation or answer-correctness metric was run. These results are descriptive evaluation outputs, not final research conclusions.

## Corrected encoder-safe rerun (separate experiment)

This section reports a second, isolated experiment. The original `results/day3/` outputs, manifest, and preceding results remain the historical 512-content-token M2 run. The corrected run writes only to `results/day3_corrected/` and `data/processed/day3_corrected/`; its metrics are not combined with the original run.

### Protocol correction and input audit

The tokenizer reported **2 special tokens** for a single input. The model sequence limit is 512, so corrected M2 uses `512 - 2 = 510` content tokens with 32-token overlap. This limit is derived at runtime from `tokenizer.num_special_tokens_to_add(pair=False)`, not hard-coded. Before indexing, the pipeline tokenizes every complete chunk with special tokens and rejects any sequence above the encoder limit.

| Method | Chunk count | Maximum encoded sequence length | Over-limit chunks | Truncation-risk chunks |
|---|---:|---:|---:|---:|
| M1 fixed 256 content | 2,263 | 259 | 0 | 0 |
| M2 fixed 510 content | 2,190 | 512 | 0 | 0 |
| M3 semantic | 2,491 | 358 | 0 | 0 |

The original M2 inventory contains **5 of 2,190 chunks** whose actual encoded sequence length is 514 (512 content tokens plus 2 special tokens). The earlier 532-token warning is produced while tokenizing a complete source document for slicing; that document tokenization is not passed to the model. The separate per-chunk audit identifies the five real encoder-limit violations in the original M2 inventory.

The corrected run uses the same seed, dataset settings, evaluation IDs, evidence, model, top-k, context budget, and query classifier settings as the original. The manifest confirms that all 200 SQuAD and 200 HotpotQA evaluation IDs match. M3 chunks were reused from the original run and checked against the corrected corpus; this preserves M3 and M4 inputs exactly. M2 alone changes its content window. The protocol version, special-token count, limits, checked chunk counts, dependency versions, and outputs are recorded in `results/day3_corrected/experiment_manifest.json`.

### Corrected held-out metrics

The corrected full run produced 1,600 rows (400 queries × four methods). Metrics below are corrected-run means. M4 continues to use exactly M3’s seed ranking for retrieval metrics.

| Method | Dataset | n | Recall@1 | Recall@3 | Recall@5 | MRR | Context precision | Context tokens | Retrieval ms | Expansion ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 fixed 256 | All | 400 | 0.614 | 0.792 | 0.858 | 0.876 | 0.247 | 584.4 | 80.71 | 0.00 |
| M2 fixed 510 content | All | 400 | 0.616 | 0.795 | 0.858 | 0.877 | 0.247 | 606.6 | 65.49 | 0.00 |
| M3 semantic | All | 400 | 0.601 | 0.773 | 0.846 | 0.865 | 0.244 | 526.3 | 60.70 | 0.00 |
| M4 adaptive | All | 400 | 0.601 | 0.773 | 0.846 | 0.865 | 0.232 | 557.4 | 60.70 | 8.53 |
| M1 fixed 256 | SQuAD | 200 | 0.820 | 0.920 | 0.950 | 0.875 | 0.190 | 675.1 | 38.52 | 0.00 |
| M2 fixed 510 content | SQuAD | 200 | 0.825 | 0.925 | 0.950 | 0.878 | 0.190 | 717.1 | 34.65 | 0.00 |
| M3 semantic | SQuAD | 200 | 0.793 | 0.900 | 0.945 | 0.856 | 0.190 | 594.7 | 35.31 | 0.00 |
| M4 adaptive | SQuAD | 200 | 0.793 | 0.900 | 0.945 | 0.856 | 0.183 | 621.5 | 35.31 | 7.59 |
| M1 fixed 256 | HotpotQA | 200 | 0.407 | 0.665 | 0.765 | 0.877 | 0.304 | 493.7 | 122.90 | 0.00 |
| M2 fixed 510 content | HotpotQA | 200 | 0.407 | 0.665 | 0.765 | 0.877 | 0.304 | 496.1 | 96.34 | 0.00 |
| M3 semantic | HotpotQA | 200 | 0.409 | 0.645 | 0.748 | 0.875 | 0.297 | 457.8 | 86.09 | 0.00 |
| M4 adaptive | HotpotQA | 200 | 0.409 | 0.645 | 0.748 | 0.875 | 0.281 | 493.3 | 86.09 | 9.48 |

Paired M4-minus-M3 bootstrap estimates in `results/day3_corrected/paired_m3_m4_bootstrap.csv` use 1,000 resamples and seed 42. Across all queries, Context Precision delta is **-0.0118** (95% percentile interval **[-0.0141, -0.0095]**), and context-token delta is **+31.1** (**[+24.8, +38.1]**). Retrieval Recall@k and MRR deltas are zero by protocol. These are descriptive estimates, not a claim that the corrected experiment is better. Corrected M2 retrieval metrics match the prior M2 aggregates on this held-out sample; its context construction also had the same aggregate token count. Query latency varies across runs and should not be interpreted as a protocol effect.

### Corrected outputs and reproduction

- Per-query CSV/JSONL, aggregate metrics, paired bootstrap estimates, plots, and manifest: `results/day3_corrected/`
- Chunk inventories and indexes: `data/processed/day3_corrected/`
- Corrected config: `configs/day3_corrected.yaml`
- Smoke command: `python -m query_adaptive_rag.evaluation.day3 --config configs/day3_corrected.yaml --smoke`
- Full command: `python -m query_adaptive_rag.evaluation.day3 --config configs/day3_corrected.yaml`

The smoke run passed before the corrected full run. The updated full test suite passed **46 tests**. The corrected full results supersede the original only for analyses that explicitly adopt the 510-content-token M2 definition; both runs remain available and separately labeled for reproducibility.
