"""Determine which chunks contribute non-duplicate text to assembled context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def context_source_chunks(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mirror ``assemble_context`` deduplication for chunk-level precision."""
    ordered = sorted(chunks, key=lambda chunk: (
        chunk["doc_id"], int(chunk["start"]), int(chunk["end"]), chunk["chunk_id"],
    ))
    contributors = []
    last_end_by_doc: dict[str, int] = {}
    seen_text: set[str] = set()
    seen_ids: set[str] = set()
    for source in ordered:
        chunk = dict(source)
        chunk_id = chunk["chunk_id"]
        key = " ".join(str(chunk["text"]).split()).casefold()
        if not key or key in seen_text or chunk_id in seen_ids:
            continue
        seen_text.add(key)
        seen_ids.add(chunk_id)
        last_end = last_end_by_doc.get(chunk["doc_id"])
        context_start = max(int(chunk["start"]), last_end or int(chunk["start"]))
        if int(chunk["end"]) <= context_start:
            continue
        chunk["context_start"] = context_start
        chunk["context_end"] = int(chunk["end"])
        contributors.append(chunk)
        last_end_by_doc[chunk["doc_id"]] = max(last_end or int(chunk["end"]), int(chunk["end"]))
    return contributors
