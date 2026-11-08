"""Experimental M4-v2: novelty-aware semantic-neighbor expansion.

This module is separate from the validated M4 implementation. It accepts the
existing M3 search response and chunk vectors, so it does not perform or alter
retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from ..chunking.text_utils import OffsetTokenizer
from ..retrieval.dense import SearchResponse
from .context_expander import ExpansionRule, TokenBudgetError, assemble_context


DEFAULT_REDUNDANCY_THRESHOLD = 0.85


@dataclass(frozen=True)
class ExpansionV2Result:
    seed_chunk_ids: list[str]
    expanded_chunk_ids: list[str]
    similarity_scores: dict[str, float]
    redundancy_scores: dict[str, float]
    context_text: str
    context_token_count: int
    token_budget: int
    latency_ms: float
    similarity_rejected_chunk_ids: list[str]
    redundancy_rejected_chunk_ids: list[str]
    budget_rejected_chunk_ids: list[str]
    deduplicated_chunk_ids: list[str]


def chunk_embedding_map_from_retriever(retriever: Any) -> dict[str, np.ndarray]:
    """Expose existing normalized M3 index vectors by stable chunk ID; no re-encoding."""
    if retriever.vectors is None:
        raise RuntimeError("build the M3 retriever before requesting its chunk embeddings")
    if len(retriever.chunks) != len(retriever.vectors):
        raise ValueError("M3 chunk rows and embedding rows are inconsistent")
    return {
        chunk["chunk_id"]: retriever.vectors[index]
        for index, chunk in enumerate(retriever.chunks)
    }


def _unit_vector(value: Any, chunk_id: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError(f"embedding for chunk {chunk_id!r} contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"embedding for chunk {chunk_id!r} has zero length")
    return vector / norm


def _neighbor_ids(chunk: Mapping[str, Any]) -> list[str]:
    return [neighbor_id for neighbor_id in (chunk.get("prev_chunk_id"), chunk.get("next_chunk_id")) if neighbor_id]


def _valid_neighbor(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    if candidate.get("doc_id") != current.get("doc_id"):
        return False
    if candidate.get("chunk_id") == current.get("chunk_id"):
        return False
    current_id = current["chunk_id"]
    return candidate.get("prev_chunk_id") == current_id or candidate.get("next_chunk_id") == current_id


def expand_context_v2(
    search: SearchResponse,
    complexity: str,
    chunks: list[dict[str, Any]],
    chunk_embeddings: Mapping[str, np.ndarray],
    tokenizer: OffsetTokenizer,
    score_chunk: Callable[[np.ndarray, str], float],
    rules: Mapping[str, ExpansionRule],
    token_budget: int,
    min_similarity: float,
    redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
) -> ExpansionV2Result:
    """Expand M3 seeds only when a relevant neighboring chunk adds new content.

    Redundancy is the maximum cosine similarity between a candidate embedding
    and any currently selected seed/expanded-chunk embedding. Scores at or
    above ``redundancy_threshold`` are rejected.
    """
    started = perf_counter()
    if complexity not in rules:
        raise ValueError(f"No expansion rule configured for complexity class {complexity!r}")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")
    if not 0.0 <= redundancy_threshold <= 1.0:
        raise ValueError("redundancy_threshold must be between 0 and 1")
    rule = rules[complexity]
    if rule.max_depth < 0 or rule.max_expansions < 0:
        raise ValueError("expansion depth and count must be non-negative")
    if not search.hits:
        raise ValueError("at least one M3 seed chunk is required")

    store = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(store) != len(chunks):
        raise ValueError("semantic chunk IDs must be unique")
    seeds = [hit.chunk for hit in search.hits]
    seed_ids = [chunk["chunk_id"] for chunk in seeds]
    if any(chunk_id not in store for chunk_id in seed_ids):
        raise ValueError("every M3 seed chunk must exist in the semantic chunk store")
    missing_vectors = [chunk_id for chunk_id in seed_ids if chunk_id not in chunk_embeddings]
    if missing_vectors:
        raise ValueError(f"M3 embedding map is missing seed chunks: {missing_vectors}")

    context, token_count = assemble_context(seeds, tokenizer)
    if token_count > token_budget:
        raise TokenBudgetError(
            f"M3 seed context needs {token_count} tokens, exceeding budget {token_budget}; "
            "increase the budget or reduce top_k"
        )

    selected = list(seeds)
    selected_ids = set(seed_ids)
    normalized_vectors = {
        chunk_id: _unit_vector(chunk_embeddings[chunk_id], chunk_id)
        for chunk_id in seed_ids
    }
    similarity_scores = {hit.chunk["chunk_id"]: float(hit.score) for hit in search.hits}
    redundancy_scores: dict[str, float] = {}
    expanded_ids: list[str] = []
    similarity_rejected: list[str] = []
    redundancy_rejected: list[str] = []
    budget_rejected: list[str] = []
    deduplicated: list[str] = []
    visited = set(seed_ids)
    frontier = list(seeds)

    for _depth in range(1, rule.max_depth + 1):
        candidates: dict[str, dict[str, Any]] = {}
        for current in frontier:
            for neighbor_id in _neighbor_ids(current):
                candidate = store.get(neighbor_id)
                if candidate is None or neighbor_id in visited or not _valid_neighbor(current, candidate):
                    continue
                visited.add(neighbor_id)
                candidates[neighbor_id] = candidate

        ranked = []
        for candidate_id, candidate in candidates.items():
            query_score = float(score_chunk(search.query_embedding, candidate_id))
            similarity_scores[candidate_id] = query_score
            if query_score < min_similarity:
                similarity_rejected.append(candidate_id)
                continue
            candidate_vector = chunk_embeddings.get(candidate_id)
            if candidate_vector is None:
                raise ValueError(f"M3 embedding map is missing candidate chunk {candidate_id!r}")
            candidate_unit = _unit_vector(candidate_vector, candidate_id)
            if candidate_unit.shape != next(iter(normalized_vectors.values())).shape:
                raise ValueError(f"embedding dimension mismatch for candidate chunk {candidate_id!r}")
            redundancy = max(
                float(np.dot(candidate_unit, selected_vector))
                for selected_vector in normalized_vectors.values()
            )
            redundancy_scores[candidate_id] = redundancy
            if redundancy >= redundancy_threshold:
                redundancy_rejected.append(candidate_id)
                continue
            ranked.append((query_score, candidate, candidate_unit))

        ranked.sort(key=lambda item: (-item[0], item[1]["doc_id"], item[1]["start"], item[1]["chunk_id"]))
        next_frontier = []
        for _query_score, candidate, candidate_vector in ranked:
            candidate_id = candidate["chunk_id"]
            if len(expanded_ids) >= rule.max_expansions:
                break
            if candidate_id in selected_ids:
                continue

            proposed_text, proposed_count = assemble_context(selected + [candidate], tokenizer)
            if proposed_text == context:
                deduplicated.append(candidate_id)
                next_frontier.append(candidate)
                continue
            if proposed_count > token_budget:
                budget_rejected.append(candidate_id)
                continue

            selected.append(candidate)
            selected_ids.add(candidate_id)
            normalized_vectors[candidate_id] = candidate_vector
            expanded_ids.append(candidate_id)
            context, token_count = proposed_text, proposed_count
            next_frontier.append(candidate)

        frontier = next_frontier
        if not frontier or len(expanded_ids) >= rule.max_expansions:
            break

    return ExpansionV2Result(
        seed_chunk_ids=seed_ids,
        expanded_chunk_ids=expanded_ids,
        similarity_scores=similarity_scores,
        redundancy_scores=redundancy_scores,
        context_text=context,
        context_token_count=token_count,
        token_budget=token_budget,
        latency_ms=(perf_counter() - started) * 1000.0,
        similarity_rejected_chunk_ids=similarity_rejected,
        redundancy_rejected_chunk_ids=redundancy_rejected,
        budget_rejected_chunk_ids=budget_rejected,
        deduplicated_chunk_ids=deduplicated,
    )
