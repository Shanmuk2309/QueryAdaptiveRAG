"""Load and normalize SQuAD and HotpotQA into a common record format."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


def _document_id(dataset: str, title: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}:{title}:{digest}"


def _squad_records(rows: Any) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        context = row["context"]
        title = row.get("title", "")
        doc_id = _document_id("squad", title, context)
        answer = row.get("answers", {})
        answer_texts = answer.get("text", [])
        answer_starts = answer.get("answer_start", [])
        evidence = []
        for text, start in zip(answer_texts, answer_starts):
            start = int(start)
            evidence.append({"doc_id": doc_id, "start": start, "end": start + len(text)})
        records.append({
            "id": str(row["id"]),
            "dataset": "squad",
            "question": row["question"],
            "answers": list(answer_texts),
            "documents": [{"doc_id": doc_id, "title": title, "text": context}],
            "gold_evidence": evidence,
            "complexity": "single-fact",
        })
    return records


def _hotpot_context_entries(context: Any, row_index: int) -> list[tuple[str, list[str]]]:
    if isinstance(context, Mapping):
        titles = context.get("title")
        sentence_groups = context.get("sentences")
        if not isinstance(titles, Sequence) or isinstance(titles, (str, bytes)):
            logger.warning("Skipping malformed HotpotQA context titles in row %d", row_index)
            return []
        if not isinstance(sentence_groups, Sequence) or isinstance(sentence_groups, (str, bytes)):
            logger.warning("Skipping malformed HotpotQA context sentences in row %d", row_index)
            return []
        if len(titles) != len(sentence_groups):
            logger.warning("HotpotQA context title/sentence counts differ in row %d", row_index)
        entries = zip(titles, sentence_groups)
    elif isinstance(context, Sequence) and not isinstance(context, (str, bytes)):
        entries = context
    else:
        logger.warning("Skipping malformed HotpotQA context in row %d", row_index)
        return []

    parsed = []
    for entry in entries:
        if isinstance(entry, Mapping):
            title, sentences = entry.get("title"), entry.get("sentences")
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:
            title, sentences = entry
        else:
            logger.warning("Ignoring malformed HotpotQA document in row %d", row_index)
            continue

        if not isinstance(title, str) or not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
            logger.warning("Ignoring malformed HotpotQA document in row %d", row_index)
            continue
        if any(not isinstance(sentence, str) for sentence in sentences):
            logger.warning("Ignoring HotpotQA document with invalid sentences in row %d", row_index)
            continue
        parsed.append((title, list(sentences)))
    return parsed


def _hotpot_supporting_facts(facts: Any, row_index: int) -> list[tuple[str, int]]:
    if isinstance(facts, Mapping):
        titles = facts.get("title")
        sentence_indices = facts.get("sent_id")
        if not isinstance(titles, Sequence) or isinstance(titles, (str, bytes)):
            logger.warning("Ignoring malformed HotpotQA supporting facts in row %d", row_index)
            return []
        if not isinstance(sentence_indices, Sequence) or isinstance(sentence_indices, (str, bytes)):
            logger.warning("Ignoring malformed HotpotQA supporting facts in row %d", row_index)
            return []
        if len(titles) != len(sentence_indices):
            logger.warning("HotpotQA supporting-fact title/index counts differ in row %d", row_index)
        entries = zip(titles, sentence_indices)
    elif isinstance(facts, Sequence) and not isinstance(facts, (str, bytes)):
        entries = facts
    else:
        logger.warning("Ignoring malformed HotpotQA supporting facts in row %d", row_index)
        return []

    parsed = []
    for entry in entries:
        if isinstance(entry, Mapping):
            title = entry.get("title")
            sentence_index = entry.get("sent_id", entry.get("sentence_index"))
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:
            title, sentence_index = entry
        else:
            logger.warning("Ignoring malformed HotpotQA supporting fact in row %d", row_index)
            continue
        if not isinstance(title, str):
            logger.warning("Ignoring HotpotQA supporting fact without a title in row %d", row_index)
            continue
        try:
            sentence_index = int(sentence_index)
        except (TypeError, ValueError):
            logger.warning("Ignoring HotpotQA supporting fact with invalid sentence index in row %d", row_index)
            continue
        parsed.append((title, sentence_index))
    return parsed


def _hotpot_records(rows: Any) -> list[dict[str, Any]]:
    records = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            logger.warning("Skipping malformed HotpotQA row %d", row_index)
            continue
        row_id = row.get("_id", row.get("id"))
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(row_id, str) or not isinstance(question, str) or not isinstance(answer, str):
            logger.warning("Skipping HotpotQA row %d with missing or invalid required fields", row_index)
            continue

        context = _hotpot_context_entries(row.get("context"), row_index)
        if not context:
            logger.warning("Skipping HotpotQA row %d without valid context documents", row_index)
            continue

        docs = []
        documents_by_title = {}
        for title, sentences in context:
            text = " ".join(str(sentence) for sentence in sentences)
            doc = {
                "doc_id": _document_id("hotpot", title, text),
                "title": title,
                "text": text,
                "sentences": sentences,
            }
            docs.append(doc)
            documents_by_title.setdefault(title, (doc, sentences))

        evidence = []
        supporting_facts = []
        for title, idx in _hotpot_supporting_facts(row.get("supporting_facts", []), row_index):
            document_entry = documents_by_title.get(title)
            if document_entry is None:
                logger.warning("Ignoring HotpotQA supporting fact for missing document %r in row %d", title, row_index)
                continue
            doc, sentences = document_entry
            if 0 <= idx < len(sentences):
                sentence = sentences[idx]
                start = sum(len(part) + 1 for part in sentences[:idx])
                supporting_facts.append({"title": title, "sentence_index": idx})
                evidence.append({"doc_id": doc["doc_id"], "start": start, "end": start + len(sentence)})
            else:
                logger.warning("Ignoring out-of-range HotpotQA supporting fact in row %d", row_index)

        records.append({
            "id": row_id,
            "dataset": "hotpotqa",
            "question": question,
            "answers": [answer],
            "documents": docs,
            "supporting_facts": supporting_facts,
            "gold_evidence": evidence,
            "complexity": "multi-hop",
        })
    return records


def load_squad(split: str = "validation", limit: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
    """Load a deterministic SQuAD slice and normalize its evidence spans."""
    from datasets import load_dataset

    rows = load_dataset("rajpurkar/squad", split=split)
    if limit is not None:
        rows = rows.shuffle(seed=seed).select(range(min(limit, len(rows))))
    return _squad_records(rows)


def load_hotpotqa(
    config: str = "distractor", split: str = "validation", limit: int | None = None, seed: int = 42
) -> list[dict[str, Any]]:
    """Load deterministic HotpotQA examples with supporting sentence spans."""
    from datasets import load_dataset

    rows = load_dataset("hotpot_qa", config, split=split)
    if limit is not None:
        rows = rows.shuffle(seed=seed).select(range(min(limit, len(rows))))
    return _hotpot_records(rows)


def load_configured_datasets(config: dict[str, dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    records = []
    for key, settings in config.items():
        if key.lower() == "squad":
            records.extend(load_squad(settings.get("split", "validation"), settings.get("limit"), seed))
        elif key.lower() in {"hotpotqa", "hotpot_qa"}:
            records.extend(load_hotpotqa(
                settings.get("config", "distractor"), settings.get("split", "validation"),
                settings.get("limit"), seed,
            ))
        else:
            raise ValueError(f"Unsupported configured dataset: {key}")
    return records
