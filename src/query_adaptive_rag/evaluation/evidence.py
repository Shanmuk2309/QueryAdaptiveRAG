"""Evidence annotation helpers for future retrieval metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def deduplicate_evidence(evidence: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return first-seen evidence spans unique by document and character offsets.

    Accepts the project's ``doc_id`` field and the more general ``document_id``
    spelling. The original annotations remain untouched; this helper returns a
    separate list suitable for metric denominators.
    """
    unique: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for item in evidence:
        document_id = item.get("document_id", item.get("doc_id"))
        if document_id is None or "start" not in item or "end" not in item:
            raise ValueError("Each evidence span must contain doc_id/document_id, start, and end")
        key = (document_id, item["start"], item["end"])
        unique.setdefault(key, dict(item))
    return list(unique.values())


def evidence_denominator(evidence: Iterable[Mapping[str, Any]]) -> int:
    """Count unique gold evidence spans for evidence-recall denominators."""
    return len(deduplicate_evidence(evidence))
