# QueryAdaptiveRAG

QueryAdaptiveRAG is a CPU-compatible research implementation for testing query-complexity-aware semantic context expansion in retrieval-augmented generation. It includes dataset preparation, fixed and semantic chunking, dense retrieval, query-only complexity classification, budgeted context expansion, and evidence-based retrieval evaluation.

The implemented complexity-aware expansion variants (M4, M4-v2, and M4-v3) share the M3 semantic retrieval seeds and add neighboring semantic chunks according to query complexity and expansion rules. M4-v3 also considers query facts when deciding whether a candidate adds useful context. All variants enforce a configured context-token budget and log candidate decisions.

## Current evaluation status

The repository contains a six-method held-out evaluation on 400 queries (200 SQuAD and 200 HotpotQA) in `results/day3_heldout_m4v3_alias_fix/`. In that run, M4-v3 had the same Recall@5, MRR, context precision, and mean context-token usage as M3 semantic retrieval, with additional expansion latency. The saved evaluation therefore does not demonstrate a retrieval or token-efficiency improvement. See `heldout_evaluation_report.md`, `aggregate_metrics.csv`, and `paired_bootstrap_ci.csv` in that directory.

The 10,000-query evaluation is configured, but its raw result directory is not currently present. Figures under `paper/figures/10000_query/` should not be treated as independently reproducible results until the corresponding per-query data and experiment manifest are available. Answer faithfulness is not evaluated; the reported metrics assess retrieval evidence coverage and context precision.

## Requirements

- Windows 11 and Python 3.10–3.12 (64-bit recommended)
- The pipeline supports CPU and CUDA through each run configuration's `device` value
- Internet access for the first download of datasets and model weights

## Setup (PowerShell)

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

If PowerShell blocks activation, set the execution policy for the current process with `Set-ExecutionPolicy -Scope Process Bypass`, then activate the environment again.

## Repair an existing environment

The pinned stack keeps NumPy 1.26.4 with PyArrow 18.1.0, Pandas 2.2.3, and scikit-learn 1.6.1. If a newer PyArrow was installed into the environment, repair these numerical packages and install Pandas' timezone dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --force-reinstall numpy==1.26.4 pyarrow==18.1.0 pandas==2.2.3 scikit-learn==1.6.1
.\.venv\Scripts\python.exe -m pip install pytz==2025.2
.\.venv\Scripts\python.exe -m pip check
```

## Run the Day 1 pipeline

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m query_adaptive_rag.pipeline --config configs/day1.yaml
```

The first run downloads the configured validation splits and embedding model. Use `--limit 5` for a small end-to-end smoke run; the limit applies to each dataset. Dataset counts, model, seed, token boundaries, and semantic chunk size/threshold settings are in `configs/day1.yaml`. Outputs are written under `data/processed/day1/`: normalized records, one JSONL file for each chunking method, and `run_manifest.json` with the effective settings, dataset/chunk counts, dependency versions, timestamp, and command. Sentence embeddings are cached under `data/cache/sentence_embeddings/`.

Each chunk record contains `doc_id`, `chunk_id`, `text`, `token_count`, character `start`/`end` offsets into its source document, `prev_chunk_id`, `next_chunk_id`, and `method`. A merged short semantic tail records its former ID in `merged_chunk_ids`. Offsets use Python Unicode character positions. Fixed chunkers use the configured Hugging Face tokenizer, overlap, and token offset mappings. Semantic chunking groups adjacent sentences using cosine similarity from L2-normalized `all-MiniLM-L6-v2` embeddings and respects the configured maximum token size. A final chunk below the configured minimum is merged with its predecessor when their combined size fits the maximum. A complete document shorter than the minimum, or a tail that would exceed the maximum if merged, remains below the minimum so no text is lost.

## Tests

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest
```

### Run the 10,000-query evaluation

The configuration `configs/day3_heldout_10000_m4v3.yaml` requests CUDA. For a CPU run, set `device: cpu` in a separate config and use separate output/cache paths. Verify CUDA availability before using the checked-in config:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

If CUDA is unavailable, use a CPU-configured copy. Preflight the dataset and chunk inputs before starting the full evaluation:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m query_adaptive_rag.evaluation.day3_heldout_10000_m4v3 --config configs\day3_heldout_10000_m4v3.yaml --preflight-only
.\.venv\Scripts\python.exe -m query_adaptive_rag.evaluation.day3_heldout_10000_m4v3 --config configs\day3_heldout_10000_m4v3.yaml --run-evaluation
```

Tests use a small deterministic fake tokenizer/model and do not download datasets or model weights.

## Day 2 retrieval and expansion smoke run

With the Day 1 smoke outputs present, run:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m query_adaptive_rag.day2_pipeline --config configs/day2_smoke.yaml
```

This builds a CPU FAISS index over the Day 1 semantic chunks, runs dense top-k retrieval for each normalized question, classifies question text with deterministic query-only rules, and writes paired semantic-baseline/adaptive context rows under `results/day2/`. The index and chunk-row mapping go under `data/processed/day2/`. The configured context budget is enforced for both methods; an over-budget seed set raises an explicit error. This is an integration smoke pipeline, not a comparative evaluation.

## M4-v3 development comparison

The existing M4-v3 development run reuses the exact M3 retrieval response and has its own historical output directory. It does not use answer/evidence labels during selection.

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m query_adaptive_rag.evaluation.day3_dev_m4v3 --config configs/day3_dev_m4v3.yaml
```

The configuration targets the existing 50 SQuAD + 50 HotpotQA development split and uses new isolated output paths under `data/processed/day3_dev_m4v3/` and `results/day3_dev_m4v3/`. Per-candidate decision reasons are written to `candidate_decisions.jsonl`. The command is a development run; it does not evaluate the held-out queries.

### M4-v3 query-focused fact expansion

The stricter fact-based implementation has a separate configuration and output directory. It accepts a neighbor only when explicit facts add an uncovered query relation or complete a supported fact chain; it has no corroboration-only acceptance path. To run it later:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m query_adaptive_rag.evaluation.day3_dev_m4v3 --config configs/day3_dev_m4v3_fact.yaml
```

Outputs go under `data/processed/day3_dev_m4v3_fact/` and `results/day3_dev_m4v3_fact/`.

## Held-out evaluation

The available 400-query held-out run is reproduced with:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m query_adaptive_rag.evaluation.day3_heldout_m4v3 --config configs/day3_heldout_m4v3_alias_fix.yaml
```

The run writes per-query and aggregate CSV/JSONL files, paired bootstrap intervals, candidate decision logs, plots, and an experiment manifest under the configured output directory. The manifest records the split identity, settings, artifact hashes, and runtime evidence controls. Do not use development results to support held-out claims.

## Reproducibility and scope

Evaluation runs record their configuration, dependency versions, data split, artifact hashes, and output paths in an experiment manifest. Keep each run's config and output directory together when sharing results. M2 is labeled fixed 510 because the encoder accepts a maximum sequence of 512 tokens and reserves two tokens for special tokens; it is not a 512-content-token chunk. Retrieval quality is measured with evidence-span Recall@K, MRR, and context precision. Faithfulness and generated-answer quality have not been evaluated.
