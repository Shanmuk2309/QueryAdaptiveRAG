# Project Plan: Query-Adaptive Semantic Chunking

## 1. Research objective

Test whether query complexity-aware expansion around retrieved semantic units can improve evidence retrieval and answer support without sending unnecessary text to an LLM. The controlled comparison has four arms: fixed 256-token chunks, fixed 512-token chunks, semantic chunks, and the proposed semantic chunks with adaptive expansion. Hold the corpus, embedding model, similarity metric, top-k seed retrieval, answer/evaluation procedure, and hardware constant. Report retrieval and context-cost outcomes; do not infer a win before running the experiment.

The measurable question is whether the proposed arm improves evidence Recall@K, MRR, and context precision at comparable or lower context-token use, with latency reported as a trade-off. Query class should be based on dataset evidence requirements (single evidence unit versus multi-evidence/multi-hop), not on the proposed classifier's own output, to avoid circular evaluation.

## 2. Lightweight architecture

```text
src/query_adaptive_rag/
  config.py             # seeds, paths, thresholds, budgets, model names
  data/                  # load, normalize, split, and validate examples
  chunking/              # fixed_size.py, semantic.py
  retrieval/             # embeddings.py, index.py, metrics.py
  expansion/             # complexity.py, context_expander.py
  evaluation/            # retrieval, answer, cost, latency measurements
  reporting/             # CSV output and plots
app.py                   # optional small Streamlit demonstration
tests/                   # unit and small integration tests
data/{raw,processed}/    # source data and generated chunks/indexes
configs/                 # versioned experiment YAML/JSON
results/                 # CSV and figures; generated, not hand edited
```

Keep chunking, retrieval, and expansion callable independently. Every indexed unit should retain document ID, unit ID, source offsets, token count, and previous/next or parent-child links. Provide one command-line experiment entry point that writes a run manifest (configuration, dependency versions, seed, timestamp) beside results.

## 3. Dependencies and CPU models

Use Python 3.10+; keep dependencies limited to `sentence-transformers` (embeddings), `faiss-cpu` (dense search), `numpy`, `pandas`, `scikit-learn` (cosine similarity / metrics), `tiktoken` or a pinned Hugging Face tokenizer for consistent token counts, `datasets` (benchmark loading), `PyYAML` (configs), `pytest` (tests), and optionally `streamlit` for a demo. Avoid LangChain unless a specific integration saves implementation time; the core algorithms are small and clearer without framework abstractions. Pin versions in a lock/requirements file after resolving dependencies.

Recommended default embedding: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, compact and CPU-friendly). `sentence-transformers/paraphrase-MiniLM-L3-v2` is a faster, lower-capacity fallback. Normalize embeddings and use cosine similarity (inner product over normalized vectors). For semantic boundaries, compute adjacent sentence embedding similarity and split when similarity falls below a fixed threshold; choose the threshold on a small development subset, then freeze it. Reuse the same sentence embeddings to build each arm's index where feasible. No model training or GPU is required.

## 4. Algorithms

### Baselines

1. **Fixed 256:** split text into tokenizer-counted 256-token windows with a fixed 32-token overlap.
2. **Fixed 512:** same policy at 512 tokens and the same 32-token overlap.
3. **Semantic:** split into sentences, group adjacent sentences while their similarity remains above the frozen threshold, and enforce minimum/maximum sizes (for example, 80/400 tokens). Oversize groups split at the nearest sentence boundary. Record all selected parameters.

Using overlap makes boundary loss less severe and must be identical in the two fixed arms. Report actual token counts; tokenizer choice is part of the configuration.

### Proposed: Complexity-Aware Semantic Context Expansion

1. Build semantic units with the exact semantic baseline segmentation.
2. Classify question complexity with transparent, deterministic features: number of explicit subquestions/conjunctions, distinct named entities, and whether the gold benchmark requires multiple supporting paragraphs. For a deployable rule, map 1 evidence target to **single-fact**, multiple related targets to **multi-concept**, and linked evidence from multiple passages to **multi-hop**. The benchmark's evidence annotations define the evaluation class; the rule's prediction can be analyzed separately.
3. Retrieve the same top-k seed units as the semantic baseline using cosine similarity. Use a fixed k (e.g., 5) in every arm.
4. Expand only within a seed's document using linked neighboring units. Add the next unit when its similarity to the query or the seed is above a frozen minimum and it adds new text. For multi-hop queries, also allow a second expansion round from selected linked units; never retrieve arbitrary neighboring documents by position alone.
5. Apply class-specific expansion limits under a shared context budget, e.g. single-fact: seed only; multi-concept: one neighbor per seed; multi-hop: up to two linked neighbors / second hop. Deduplicate overlapping spans and order selected evidence by retrieval score, preserving source order within a document. Enforce the same maximum context token budget for all approaches.

The distinguishing mechanism is **query-conditioned allocation of neighboring semantic units after common seed retrieval**, with evidence-relationship links and a hard token budget. Semantic chunking alone changes segmentation; this method keeps semantic segmentation fixed and varies expansion depth by query class. Treat this as a project hypothesis/contribution, not a claim of first-ever novelty; position it against established semantic segmentation and multi-hop retrieval work after a literature review.

## 5. Dataset and experiment protocol

Use a modest, documented subset to fit the schedule: HotpotQA distractor/dev examples for annotated multi-hop evidence, plus a single-fact QA set (e.g. SQuAD v1.1 dev) to test the single-fact class. Fix and publish dataset versions, sample IDs, and a seeded stratified sample (for example, 200 examples per class if available). Use source passages as the corpus; keep question, answer, evidence document/sentence IDs, and class label in each example. A normalized JSONL record can contain `id`, `question`, `answer`, `complexity`, `documents:[{doc_id,text}]`, and `gold_evidence:[{doc_id,start,end}]`. Do not use test labels to tune thresholds.

Run all four arms over the same questions and corpus with one embedding model, identical top-k and token budget, same token counter, and fixed random seed. Freeze semantic threshold and expansion rules on a development slice before evaluation. Save one row per query and method to CSV, including config/run ID, class, Recall@K, reciprocal rank, context precision, context tokens, latency, and answer/evidence outcomes. Report macro averages and per-complexity results; include paired bootstrap confidence intervals if time permits. Compare context use alongside quality, not as a standalone success metric. Re-run a small deterministic subset to check repeatability.

For answer faithfulness, use the same frozen generator and prompt across arms if CPU/runtime allows, with answers checked against gold answers and cited evidence. Otherwise report retrieval evidence support as the primary result and clearly mark faithfulness as not evaluated; do not substitute an unvalidated LLM-as-judge score. Retrieval latency must state whether embedding/index construction is excluded and report query-time separately.

## 6. Challenges and mitigations

- **Unequal chunk counts or context sizes:** preserve common top-k, cap total tokens, and report both retrieved units and tokens.
- **Complexity-label noise:** use gold evidence structure for evaluation labels; report rule-classifier agreement separately.
- **Threshold sensitivity:** tune only on development data and include a small threshold sensitivity analysis.
- **Tokenization mismatch:** pin the tokenizer and count the final assembled context with that same tokenizer.
- **Dataset licensing/scale:** record source, version, license, subset IDs, and avoid downloading the full corpus unnecessarily.
- **CPU latency:** cache embeddings, use FAISS flat search at small scale, batch encoding, and separate build time from query time.
- **Overstated conclusions:** run paired comparisons, disclose sample size and limitations, and do not claim statistical significance without uncertainty estimates.

## 7. Three-day implementation and paper schedule

### Day 1 — Data and baselines

- Confirm dataset subset, labels, token counter, model ID, config, and random seed; record environment versions.
- Build data validation and normalized JSONL loading, sentence/token utilities, and the two fixed chunkers plus semantic segmenter.
- Add chunker unit tests for boundaries, overlap, metadata, and token limits.
- Generate chunks and cached embeddings; verify counts and inspect a small sample manually.
- Paper: draft introduction, research question, related-work outline, and method overview; make no results claims.

### Day 2 — Retrieval, proposed method, and evaluation

- Implement a shared cosine/FAISS index path and seed retrieval for all methods.
- Implement deterministic complexity features and adaptive linked-unit expansion with configurable class limits and hard budget.
- Add metric code and tests for Recall@K, MRR, context precision, token count, and expansion budget enforcement.
- Run a small smoke experiment, fix errors, freeze thresholds/configuration, then run the full seeded subset and save per-query CSV plus run manifest.
- Paper: finalize dataset, methods, and evaluation protocol; create result tables from actual CSV only.

### Day 3 — Analysis, reproducibility, and writing

- Check missing/duplicate queries, inspect failure cases by class, recompute aggregate/per-class metrics, and run a small repeatability check.
- Produce plots/tables, document hardware, versions, latency timing boundaries, and any unevaluated metrics.
- Write results, limitations, discussion, and conclusion strictly from measured outcomes; add citations and verify them.
- Run the full test suite, capture one optional Streamlit demo, clean the repository, and ensure configs, sample IDs, CSVs, and commands reproduce the reported experiment.

## 8. Literature anchors

- Hearst, *TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages* (1997), a foundational topic segmentation method: [ACL Anthology](https://aclanthology.org/J97-1003/).
- Reimers & Gurevych, *Sentence-BERT* (2019), sentence embeddings for semantic similarity: [ACL Anthology](https://aclanthology.org/D19-1410/).
- Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering* (2018): [ACL Anthology](https://aclanthology.org/D18-1259/).
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2024), motivation for measuring context length and evidence placement: [TACL](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long).
