"""Budgeted semantic-neighbor expansion with explicit provenance."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from ..chunking.text_utils import OffsetTokenizer, count_tokens
from ..retrieval.dense import SearchResponse


class TokenBudgetError(ValueError):
    """Raised when the required seed context cannot fit the hard token budget."""


@dataclass(frozen=True)
class ExpansionRule:
    max_depth: int
    max_expansions: int


@dataclass(frozen=True)
class ExpansionResult:
    seed_chunk_ids: list[str]
    expanded_chunk_ids: list[str]
    similarity_scores: dict[str, float]
    context_text: str
    context_token_count: int
    token_budget: int
    latency_ms: float
    budget_rejected_chunk_ids: list[str]
    deduplicated_chunk_ids: list[str]


def assemble_context(chunks: list[dict[str, Any]], tokenizer: OffsetTokenizer) -> tuple[str, int]:
    """Render chunks in document/source order, removing overlap and duplicate text."""
    ordered = sorted(chunks, key=lambda c: (c["doc_id"], c["start"], c["end"], c["chunk_id"]))
    pieces: list[str] = []
    last_end_by_doc: dict[str, int] = {}
    text_keys: set[str] = set()
    for chunk in ordered:
        text = chunk["text"]
        key = " ".join(text.split()).casefold()
        if not key or key in text_keys:
            continue
        text_keys.add(key)
        last_end = last_end_by_doc.get(chunk["doc_id"])
        if last_end is not None and chunk["start"] < last_end:
            overlap = min(last_end - chunk["start"], len(text))
            text = text[overlap:]
        if not text.strip():
            continue
        pieces.append(text)
        last_end_by_doc[chunk["doc_id"]] = max(last_end or chunk["end"], chunk["end"])
    rendered = "\n\n".join(pieces)
    return rendered, count_tokens(rendered, tokenizer)


def _neighbor_ids(chunk: Mapping[str, Any]) -> list[str]:
    return [candidate for candidate in (chunk.get("prev_chunk_id"), chunk.get("next_chunk_id")) if candidate]


def _is_valid_neighbor(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    if candidate.get("doc_id") != current.get("doc_id") or candidate.get("chunk_id") == current.get("chunk_id"):
        return False
    current_id = current["chunk_id"]
    reciprocal = candidate.get("next_chunk_id") == current_id or candidate.get("prev_chunk_id") == current_id
    return reciprocal


def expand_context(
    search: SearchResponse,
    complexity: str,
    chunks: list[dict[str, Any]],
    tokenizer: OffsetTokenizer,
    score_chunk: Callable[[np.ndarray, str], float],
    rules: Mapping[str, ExpansionRule],
    token_budget: int,
    min_similarity: float,
) -> ExpansionResult:
    """Expand the unchanged top-k seeds along relevant same-document neighbors."""
    started = perf_counter()
    if complexity not in rules:
        raise ValueError(f"No expansion rule configured for complexity class {complexity!r}")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")
    rule = rules[complexity]
    if rule.max_depth < 0 or rule.max_expansions < 0:
        raise ValueError("expansion depth and count must be non-negative")
    if not search.hits:
        raise ValueError("at least one seed chunk is required")

    store = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(store) != len(chunks):
        raise ValueError("semantic chunk IDs must be unique")
    seeds = [hit.chunk for hit in search.hits]
    seed_ids = [chunk["chunk_id"] for chunk in seeds]
    if any(chunk_id not in store for chunk_id in seed_ids):
        raise ValueError("every seed chunk must exist in the semantic chunk store")
    context, token_count = assemble_context(seeds, tokenizer)
    if token_count > token_budget:
        raise TokenBudgetError(
            f"top-k seed context needs {token_count} tokens, exceeding budget {token_budget}; "
            "increase the budget or reduce top_k"
        )

    selected = list(seeds)
    selected_ids = set(seed_ids)
    scores = {hit.chunk["chunk_id"]: float(hit.score) for hit in search.hits}
    expanded_ids: list[str] = []
    rejected_ids: list[str] = []
    deduplicated_ids: list[str] = []
    visited = set(seed_ids)
    frontier = list(seeds)

    for _depth in range(1, rule.max_depth + 1):
        candidates: dict[str, dict[str, Any]] = {}
        for current in frontier:
            for neighbor_id in _neighbor_ids(current):
                candidate = store.get(neighbor_id)
                if candidate is None or neighbor_id in visited or not _is_valid_neighbor(current, candidate):
                    continue
                visited.add(neighbor_id)
                candidates[neighbor_id] = candidate

        ranked = []
        for candidate_id, candidate in candidates.items():
            score = float(score_chunk(search.query_embedding, candidate_id))
            scores[candidate_id] = score
            if score >= min_similarity:
                ranked.append((score, candidate))
        ranked.sort(key=lambda item: (-item[0], item[1]["doc_id"], item[1]["start"], item[1]["chunk_id"]))

        next_frontier = []
        for _score, candidate in ranked:
            candidate_id = candidate["chunk_id"]
            if len(expanded_ids) >= rule.max_expansions:
                break
            if candidate_id in selected_ids:
                continue

            proposed = selected + [candidate]
            proposed_text, proposed_count = assemble_context(proposed, tokenizer)
            if proposed_text == context:
                deduplicated_ids.append(candidate_id)
                next_frontier.append(candidate)
                continue
            if proposed_count > token_budget:
                rejected_ids.append(candidate_id)
                continue

            selected.append(candidate)
            selected_ids.add(candidate_id)
            expanded_ids.append(candidate_id)
            context, token_count = proposed_text, proposed_count
            next_frontier.append(candidate)
        frontier = next_frontier
        if not frontier or len(expanded_ids) >= rule.max_expansions:
            break

    return ExpansionResult(
        seed_chunk_ids=seed_ids,
        expanded_chunk_ids=expanded_ids,
        similarity_scores=scores,
        context_text=context,
        context_token_count=token_count,
        token_budget=token_budget,
        latency_ms=(perf_counter() - started) * 1000.0,
        budget_rejected_chunk_ids=rejected_ids,
        deduplicated_chunk_ids=deduplicated_ids,
    )
