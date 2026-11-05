# Day 2 Implementation Audit

## Scope and implementation

Day 2 adds dense retrieval, query-only complexity classification, and budgeted semantic-neighbor expansion. It reuses the existing Day 1 normalized JSONL, semantic chunks, tokenizer utilities, settings, and evidence spans. Day 1 fixed chunkers and normalized outputs were not modified. No retrieval metrics, answer generation, final comparisons, or paper conclusions were produced.

- `retrieval/dense.py` encodes chunks with `sentence-transformers/all-MiniLM-L6-v2`, normalizes vectors, builds/saves a FAISS `IndexFlatIP` CPU index, and returns top-k chunks with stable IDs, cosine scores, query embeddings, and query-time latency. Ties are sorted by score then chunk ID.
- `classification/complexity.py` accepts only a query string. It reports `single_fact`, `multi_concept`, or `multi_hop`, plus entity, conjunction, comparison, relation, and rule-reason features. It does not receive answers, gold evidence, or dataset labels.
- `expansion/context_expander.py` takes the exact `SearchResponse` seeds used by the semantic baseline. Configured class policies traverse reciprocal previous/next links within the same document, score candidates against the query vector, filter by a fixed similarity threshold, deduplicate repeated/overlapping text, and render in source order. It retains seed and added chunk IDs separately. It rejects additions that exceed the hard budget and records rejected IDs; if the seeds alone exceed the budget, it raises `TokenBudgetError` instead of truncating silently.
- `day2_pipeline.py` writes the FAISS index and row-to-chunk map under `data/processed/day2/`, and paired per-query CSV/JSONL rows under `results/day2/`. The run manifest records the effective config, command, seed, model/tokenizer, dependency versions, and input/output counts.

The smoke policy is explicit in `configs/day2_smoke.yaml`: seed top-k 5, 2,048-token budget, minimum neighbor similarity 0.20; single-fact allows no neighbors, multi-concept up to two at depth one, and multi-hop up to four at depth two. These settings were not tuned on gold evidence.

## Tests and smoke run

Full test command:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest -q
```

**Result:** 33 tests passed. Coverage includes FAISS top-k/tie ordering and IDs; each complexity class and query-only classifier input; same-document reciprocal neighbor constraints; relevance filtering; source order and text deduplication; token-budget rejection and explicit seed-budget errors; and seed equality between baseline and proposed methods.

Smoke command:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m query_adaptive_rag.day2_pipeline --config configs/day2_smoke.yaml
```

The pipeline completed on the Day 1 smoke corpus: 10 queries and 63 semantic chunks yielded 10 semantic-baseline rows and 10 adaptive-expansion rows. All ten paired rows have identical seed IDs. The saved FAISS index contains 63 vectors of dimension 384. Every logged context is within 2,048 tokens; no candidate was rejected by the budget. One neighbor was added across this small run. These are pipeline counts and behavior checks, not retrieval-quality results.

Generated files:

- `data/processed/day2/semantic.faiss`
- `data/processed/day2/index_chunks.jsonl`
- `data/processed/day2/run_manifest.json`
- `results/day2/per_query.jsonl`
- `results/day2/per_query.csv`

## Smoke-set observations and limitations

- Runtime classifications on this ten-query smoke sample were three `single_fact`, seven `multi_concept`, and zero `multi_hop`. Unit tests cover the multi-hop rule, but this corpus slice does not exercise it end to end.
- The classifier uses transparent capitalization and cue-word heuristics. Entity extraction and complexity predictions can be noisy; no classifier accuracy is claimed.
- The neighbor similarity threshold and expansion budgets are fixed illustrative settings, not tuned or validated against gold evidence. The sample has only one accepted expansion, so it cannot establish a quality or token-efficiency benefit.
- Retrieval latency includes query embedding and FAISS search; corpus embedding/index construction is excluded. Expansion latency includes candidate scoring and context rendering.
- The run is a smoke/integration check only. Recall, MRR, context precision, faithfulness, and comparative statistical analysis remain unimplemented for the later evaluation stage.
- Hugging Face's tokenizer emitted a non-fatal 638-token-versus-512 warning while counting a long assembled context. Token counting did not truncate the text: the logged contexts remained within the configured 2,048-token budget. Semantic chunk and question inputs to the retriever are shorter than 512 tokens in this sample.
