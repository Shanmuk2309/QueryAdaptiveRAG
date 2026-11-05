# Day 1 Output Audit

**Audited:** `data/processed/day1/` and `data/cache/sentence_embeddings/`  
**Run:** `python -m query_adaptive_rag.pipeline --config configs/day1.yaml --limit 5`  
**Scope:** read-only audit of regenerated Day 1 outputs after the maintenance pass. No Day 2 features were added.

## Output inventory and counts

All five files in `data/processed/day1/` were read successfully. All JSONL rows parsed as UTF-8 JSON.

| File | Records |
| --- | ---: |
| `normalized.jsonl` | 10 examples total: 5 SQuAD and 5 HotpotQA |
| `fixed_256_chunks.jsonl` | 58 chunks |
| `fixed_512_chunks.jsonl` | 55 chunks |
| `semantic_chunks.jsonl` | 63 chunks |
| `run_manifest.json` | 1 run record |

The CLI limit applies separately to each dataset. There are 55 unique source documents. Fixed-size outputs did not change during this pass.

## Run manifest

The manifest records the effective five-example limit for each dataset, seed `42`, effective configuration, UTC timestamp, Python executable and platform, executed command, model/tokenizer identifiers, dataset and chunk counts, unique document count, and installed versions for the project's relevant dependencies (including pytest). It is generated automatically after each successful pipeline run.

## Embedding cache

The cache contains 55 `.npy` files and 294 sentence vectors total. Every array has shape `(sentence_count, 384)`; all values are finite and vector norms are approximately 1.0 (range `0.99999988`–`1.00000012`). The embeddings match the 384-dimensional `all-MiniLM-L6-v2` setup and are cached outside the processed-output directory as configured.

## Chunk metadata and relationships

- All 176 chunks have unique IDs within their method file, document IDs found in normalized records, positive token counts, source offsets, and previous/next relationship fields.
- For every chunk, slicing its referenced document at `[start:end]` reproduces its text exactly. Retokenizing with the configured fast tokenizer matches every stored token count.
- Neighbor order and reciprocal links are valid for every chunk.
- Fixed 256-token chunks range from 43 to 256 tokens; fixed 512-token chunks range from 43 to 333 tokens. The smaller observed maximum for the 512-token method reflects the short sampled passages.
- Semantic chunks range from 43 to 214 tokens, with no chunks above the 400-token maximum. Twenty-three short trailing chunks were merged into their preceding chunk; the predecessor's chunk ID stayed stable and each folded ID is retained in `merged_chunk_ids`.

## Gold evidence and deduplication

- All five SQuAD examples retain 16 answer-span annotations, representing 7 unique `(doc_id, start, end)` spans. Duplicate annotations remain in normalized data to preserve the source. The reusable `deduplicate_evidence` / `evidence_denominator` utilities return seven unique spans for denominator calculations.
- All five HotpotQA examples retain both normalized `supporting_facts` title/sentence-index pairs and `gold_evidence` character spans: 17 of each, all unique in this sample. Every gold span maps to a document and valid source offsets.
- Tests confirm duplicate annotations do not inflate the deduplicated denominator and that normalization preserves original evidence annotations.

## Remaining limitations

- Twelve semantic chunks are complete source documents shorter than the configured 80-token minimum. They remain intact because they have no preceding chunk to merge with. No short trailing chunk remained in this smoke run after eligible merges. On other data, a short tail also remains when merging it would exceed the 400-token maximum. No text or evidence is discarded in these cases.
- This was a five-example-per-dataset smoke run, not a retrieval experiment. It produces no retrieval metrics, indexes, evaluation CSV, or answer-generation results.
