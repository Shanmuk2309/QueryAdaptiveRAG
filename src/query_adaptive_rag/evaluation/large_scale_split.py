"""Reproducible expansion of the frozen Day 3 held-out query split."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


DATASETS = ("squad", "hotpotqa")
TARGET_PER_DATASET = 5_000
DEV_PER_DATASET = 50
BASE_HELDOUT_PER_DATASET = 200
BASE_POOL_PER_DATASET = 250


def ordered_pairs_hash(pairs: Sequence[tuple[str, str]]) -> str:
    payload = json.dumps(list(pairs), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def id_list_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def extend_frozen_split(
    shuffled_ids: Mapping[str, Sequence[str]],
    development_ids: Mapping[str, Sequence[str]],
    heldout_ids: Mapping[str, Sequence[str]],
    target_per_dataset: int = TARGET_PER_DATASET,
) -> dict[str, Any]:
    """Preserve frozen IDs and append unused IDs in the seeded source order."""
    if target_per_dataset < BASE_HELDOUT_PER_DATASET:
        raise ValueError("target cannot be smaller than the frozen held-out split")
    splits: dict[str, Any] = {}
    ordered_pairs: list[tuple[str, str]] = []
    for dataset in DATASETS:
        ordered = [str(value) for value in shuffled_ids[dataset]]
        dev = [str(value) for value in development_ids[dataset]]
        frozen_test = [str(value) for value in heldout_ids[dataset]]
        if len(dev) != DEV_PER_DATASET or len(set(dev)) != len(dev):
            raise ValueError(f"{dataset} must have {DEV_PER_DATASET} unique development IDs")
        if len(frozen_test) != BASE_HELDOUT_PER_DATASET or len(set(frozen_test)) != len(frozen_test):
            raise ValueError(f"{dataset} must have {BASE_HELDOUT_PER_DATASET} unique frozen held-out IDs")
        if set(dev) & set(frozen_test):
            raise ValueError(f"{dataset} development and held-out IDs overlap")
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"{dataset} source ID order contains duplicate IDs")
        source_set = set(ordered)
        reserved = set(dev) | set(frozen_test)
        if not reserved <= source_set:
            raise ValueError(f"{dataset} frozen IDs are absent from the official validation source")
        if len(ordered) < target_per_dataset + len(dev):
            raise ValueError(f"{dataset} has too few source IDs for the requested balanced held-out size")
        additions = [query_id for query_id in ordered if query_id not in reserved]
        additions = additions[:target_per_dataset - len(frozen_test)]
        evaluation = frozen_test + additions
        if len(evaluation) != target_per_dataset or len(set(evaluation)) != target_per_dataset:
            raise ValueError(f"{dataset} expanded held-out ID list is not the requested unique size")
        if set(dev) & set(evaluation):
            raise ValueError(f"{dataset} development and expanded held-out IDs overlap")
        splits[dataset] = {
            "source_split": "validation",
            "source_record_count": len(ordered),
            "load_seed": 42 if dataset == "squad" else 43,
            "development_ids": dev,
            "development_ids_sha256": id_list_hash(dev),
            "frozen_heldout_ids": frozen_test,
            "additional_heldout_ids": additions,
            "evaluation_ids": evaluation,
            "evaluation_ids_sha256": id_list_hash(evaluation),
            "development_count": len(dev),
            "frozen_heldout_count": len(frozen_test),
            "additional_heldout_count": len(additions),
            "evaluation_count": len(evaluation),
        }
        ordered_pairs.extend((dataset, query_id) for query_id in evaluation)
    if len(ordered_pairs) != target_per_dataset * len(DATASETS):
        raise ValueError("expanded split has an unexpected total query count")
    return {
        "schema_version": 1,
        "selection_method": "existing loader shuffle seed; preserve frozen IDs; append first unused IDs in shuffled source order",
        "target_per_dataset": target_per_dataset,
        "total_evaluation_count": len(ordered_pairs),
        "datasets": splits,
        "ordered_dataset_query_ids": [[dataset, query_id] for dataset, query_id in ordered_pairs],
        "ordered_dataset_query_ids_sha256": ordered_pairs_hash(ordered_pairs),
    }


def validate_expanded_split(
    manifest: Mapping[str, Any],
    development_reference: Mapping[str, Sequence[str]],
    original_reference: Mapping[str, Sequence[str]],
    corrected_reference: Mapping[str, Sequence[str]],
    target_per_dataset: int = TARGET_PER_DATASET,
) -> str:
    """Check exact counts, frozen-ID preservation, order, and ID-set hashes."""
    pair_list: list[tuple[str, str]] = []
    datasets = manifest.get("datasets", {})
    for dataset in DATASETS:
        entry = datasets.get(dataset, {})
        dev = [str(value) for value in entry.get("development_ids", [])]
        evaluation = [str(value) for value in entry.get("evaluation_ids", [])]
        frozen = [str(value) for value in entry.get("frozen_heldout_ids", [])]
        additions = [str(value) for value in entry.get("additional_heldout_ids", [])]
        if dev != list(development_reference[dataset]):
            raise ValueError(f"{dataset} development IDs changed")
        if frozen != list(original_reference[dataset]) or frozen != list(corrected_reference[dataset]):
            raise ValueError(f"{dataset} frozen held-out IDs/order changed")
        if evaluation != frozen + additions:
            raise ValueError(f"{dataset} expanded test does not preserve frozen ID order")
        if len(evaluation) != target_per_dataset or len(set(evaluation)) != target_per_dataset:
            raise ValueError(f"{dataset} must contain exactly {target_per_dataset} unique held-out IDs")
        if len(dev) != DEV_PER_DATASET or len(set(dev)) != DEV_PER_DATASET:
            raise ValueError(f"{dataset} development set must contain exactly {DEV_PER_DATASET} unique IDs")
        if set(dev) & set(evaluation):
            raise ValueError(f"{dataset} development/held-out overlap")
        if len(additions) != target_per_dataset - len(frozen):
            raise ValueError(f"{dataset} additional held-out count is incorrect")
        if entry.get("evaluation_ids_sha256") != id_list_hash(evaluation):
            raise ValueError(f"{dataset} evaluation ID hash mismatch")
        pair_list.extend((dataset, query_id) for query_id in evaluation)
    expected_pairs = [[dataset, query_id] for dataset, query_id in pair_list]
    if manifest.get("ordered_dataset_query_ids") != expected_pairs:
        raise ValueError("ordered dataset/query ID list mismatch")
    digest = ordered_pairs_hash(pair_list)
    if manifest.get("ordered_dataset_query_ids_sha256") != digest:
        raise ValueError("ordered dataset/query ID hash mismatch")
    if len(pair_list) != target_per_dataset * 2 or len(set(pair_list)) != len(pair_list):
        raise ValueError("expanded split does not contain the expected unique total")
    return digest
