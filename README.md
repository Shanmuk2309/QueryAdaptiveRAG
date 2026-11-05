# QueryAdaptiveRAG

Day 1 provides normalized SQuAD/HotpotQA loading and three chunking outputs for the planned four-method study: fixed 256-token, fixed 512-token, and semantic. The proposed query-aware expansion method is not implemented yet. No experimental results are included.

## Requirements

- Windows 11 and Python 3.10–3.12 (64-bit recommended)
- CPU only; the pipeline explicitly loads `all-MiniLM-L6-v2` on CPU
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

The pinned stack keeps NumPy 1.26.4 with PyArrow 18.1.0, Pandas 2.2.3, and scikit-learn 1.6.1. If a newer PyArrow was installed into the environment, repair just these numerical packages and install Pandas' timezone dependency:

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

Each chunk record contains `doc_id`, stable-within-run `chunk_id`, `text`, `token_count`, character `start`/`end` offsets into its source document, `prev_chunk_id`, `next_chunk_id`, and `method`. A merged short semantic tail records its former ID in `merged_chunk_ids`. Offsets use Python Unicode character positions. Fixed chunkers use the configured Hugging Face tokenizer, 32-token overlap, and exact token offset mappings. Semantic chunking groups adjacent sentences using cosine similarity from L2-normalized `all-MiniLM-L6-v2` embeddings and respects the configured maximum token size. A final chunk below the configured minimum is merged with its predecessor when their combined size fits the maximum. A complete document shorter than the minimum, or a tail that would exceed the maximum if merged, remains below the minimum so no text is lost.

## Tests

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest
```

Tests use a small deterministic fake tokenizer/model and do not download datasets or model weights.

## Reproducibility and scope

The configuration seeds Python and NumPy and deterministically shuffles each selected dataset slice. Record the config and dependency versions with future experiment results. This pipeline only prepares data and chunks; retrieval indexing, adaptive context expansion, answer generation, and evaluation are later project stages.
