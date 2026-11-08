# Six-Method Held-Out Evaluation

- Run ID: `day3_heldout_m4v3_alias_fix_20261106`
- Held-out queries: 400 (200 SQuAD, 200 HotpotQA)
- Ordered query ID hash: `762bd07ae47cc0a750046e04c1eb64697875b0643e431cc0af3cee6b5adf8bab`
- M3 semantic chunks: 2491

## Aggregate results

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Context precision | Mean context tokens | Expansion latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1_fixed_256 | 0.6135 | 0.7923 | 0.8577 | 0.8759 | 0.2470 | 584.41 | 0.000 |
| M2_fixed_510 | 0.6160 | 0.7948 | 0.8577 | 0.8773 | 0.2470 | 606.62 | 0.000 |
| M3_semantic | 0.6006 | 0.7727 | 0.8465 | 0.8653 | 0.2435 | 526.25 | 0.000 |
| M4_adaptive | 0.6006 | 0.7727 | 0.8465 | 0.8653 | 0.2317 | 557.36 | 4.579 |
| M4_v2_redundancy | 0.6006 | 0.7727 | 0.8465 | 0.8653 | 0.2321 | 556.69 | 4.869 |
| M4_v3_fact_alias_fix | 0.6006 | 0.7727 | 0.8465 | 0.8653 | 0.2435 | 526.25 | 9.752 |

## Dataset-level results

| Method | Dataset | Queries | Context precision | Mean context tokens | Expansion latency (ms) |
|---|---|---:|---:|---:|---:|
| M1_fixed_256 | hotpotqa | 200 | 0.3040 | 493.68 | 0.000 |
| M1_fixed_256 | squad | 200 | 0.1900 | 675.14 | 0.000 |
| M2_fixed_510 | hotpotqa | 200 | 0.3040 | 496.13 | 0.000 |
| M2_fixed_510 | squad | 200 | 0.1900 | 717.10 | 0.000 |
| M3_semantic | hotpotqa | 200 | 0.2970 | 457.77 | 0.000 |
| M3_semantic | squad | 200 | 0.1900 | 594.74 | 0.000 |
| M4_adaptive | hotpotqa | 200 | 0.2806 | 493.26 | 4.287 |
| M4_adaptive | squad | 200 | 0.1827 | 621.47 | 4.870 |
| M4_v2_redundancy | hotpotqa | 200 | 0.2814 | 491.92 | 4.527 |
| M4_v2_redundancy | squad | 200 | 0.1827 | 621.47 | 5.212 |
| M4_v3_fact_alias_fix | hotpotqa | 200 | 0.2970 | 457.77 | 9.434 |
| M4_v3_fact_alias_fix | squad | 200 | 0.1900 | 594.74 | 10.070 |

## Runtime and evidence controls

M4, M4-v2, and M4-v3 used the same M3 search response for each query. Gold evidence was passed to metric calculation only after contexts for all held-out queries were selected. No answer-generation metric was substituted for evidence retrieval.

## Candidate decisions

Per-query, per-method candidate decisions and acceptance/rejection reasons are in `candidate_decisions.jsonl`.

All generated contexts were checked against the configured hard token budget before metric calculation.
