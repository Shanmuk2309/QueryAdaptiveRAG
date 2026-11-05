"""Evidence retrieval metrics with explicit span-overlap semantics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .evidence import deduplicate_evidence


def chunk_matches_evidence(chunk: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    """True when chunk and evidence have the same document and overlapping offsets."""
    document_id = evidence.get("document_id", evidence.get("doc_id"))
    chunk_start = chunk.get("context_start", chunk.get("start", 0))
    chunk_end = chunk.get("context_end", chunk.get("end", 0))
    return (
        chunk.get("doc_id") == document_id
        and int(chunk_start) < int(evidence["end"])
        and int(evidence["start"]) < int(chunk_end)
    )


def recall_at_k(
    ranked_chunks: Sequence[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]], k: int
) -> float:
    """Unique gold evidence spans covered by the first k ranked chunks / total spans."""
    unique_evidence = deduplicate_evidence(evidence)
    if not unique_evidence:
        raise ValueError("recall is undefined for a query with no gold evidence")
    if k <= 0:
        raise ValueError("k must be positive")
    retrieved = ranked_chunks[:k]
    covered = sum(any(chunk_matches_evidence(chunk, span) for chunk in retrieved) for span in unique_evidence)
    return covered / len(unique_evidence)


def reciprocal_rank(
    ranked_chunks: Sequence[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]]
) -> float:
    """Reciprocal rank of the first chunk matching any unique gold evidence span."""
    unique_evidence = deduplicate_evidence(evidence)
    if not unique_evidence:
        raise ValueError("MRR is undefined for a query with no gold evidence")
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if any(chunk_matches_evidence(chunk, span) for span in unique_evidence):
            return 1.0 / rank
    return 0.0


def context_precision(
    context_chunks: Sequence[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]]
) -> float:
    """Relevant chunks that contribute text / chunks that contribute text."""
    unique_evidence = deduplicate_evidence(evidence)
    if not unique_evidence:
        raise ValueError("context precision is undefined for a query with no gold evidence")
    unique_chunks = _context_contributors(context_chunks)
    if not unique_chunks:
        return 0.0
    relevant = sum(
        any(chunk_matches_evidence(chunk, span) for span in unique_evidence)
        for chunk in unique_chunks.values()
    )
    return relevant / len(unique_chunks)


def _context_contributors(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Mirror context rendering by excluding exact duplicates and fully covered spans."""
    ordered = sorted(chunks, key=lambda c: (c["doc_id"], c["start"], c["end"], c["chunk_id"]))
    contributors: dict[str, Mapping[str, Any]] = {}
    last_end: dict[str, int] = {}
    seen_text: set[str] = set()
    for chunk in ordered:
        key = " ".join(str(chunk["text"]).split()).casefold()
        if not key or key in seen_text:
            continue
        seen_text.add(key)
        covered_end = last_end.get(chunk["doc_id"])
        if covered_end is not None and int(chunk["end"]) <= covered_end:
            continue
        contributors[chunk["chunk_id"]] = chunk
        last_end[chunk["doc_id"]] = max(covered_end or int(chunk["end"]), int(chunk["end"]))
    return contributors


def calculate_metrics(
    ranked_chunks: Sequence[Mapping[str, Any]],
    context_chunks: Sequence[Mapping[str, Any]],
    evidence: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    evidence = deduplicate_evidence(evidence)
    return {
        "recall_at_1": recall_at_k(ranked_chunks, evidence, 1),
        "recall_at_3": recall_at_k(ranked_chunks, evidence, 3),
        "recall_at_5": recall_at_k(ranked_chunks, evidence, 5),
        "mrr": reciprocal_rank(ranked_chunks, evidence),
        "context_precision": context_precision(context_chunks, evidence),
    }
