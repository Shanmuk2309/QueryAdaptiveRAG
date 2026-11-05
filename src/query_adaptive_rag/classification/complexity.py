"""Deterministic, interpretable query-complexity heuristics.

Only the query string is accepted. Gold evidence, answer text, and dataset labels
are deliberately absent from the classifier interface.
"""

from __future__ import annotations

import re
from typing import Any


_CONJUNCTION = re.compile(r"\b(?:and|also|as well as|or|both)\b", re.IGNORECASE)
_COMPARISON = re.compile(
    r"\b(?:compare|compared|versus|vs\.?|difference|similar|unlike|between|more than|less than)\b",
    re.IGNORECASE,
)
_RELATIONS = re.compile(
    r"\b(?:director|directed|writer|wrote|author|written|starred|stars|played|portrayed|"
    r"located|based on|adapted from|founded|created|invented|member of|part of|capital of|"
    r"spouse|wife of|husband of|parent of|son of|daughter of|related to|connected to|"
    r"influenced|influence|produced|producer|coach|teammate|born in|died in)\b",
    re.IGNORECASE,
)
_NESTED_RELATION = re.compile(
    r"\b(?:who|what|which)\b.{0,100}\b(?:that|which|who|whose)\b.{0,100}\b(?:directed|wrote|starred|"
    r"played|founded|created|invented|produced|located|based)\b",
    re.IGNORECASE,
)
_ENTITY_PHRASE = re.compile(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*")
_QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "compare", "tell", "name", "list", "identify", "describe", "explain", "show", "give", "find",
}


def classify_query(query: str) -> dict[str, Any]:
    """Return a class and the query-only features that led to it."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    conjunctions = [match.group(0).lower() for match in _CONJUNCTION.finditer(query)]
    comparisons = [match.group(0).lower() for match in _COMPARISON.finditer(query)]
    relation_matches = [match.group(0).lower() for match in _RELATIONS.finditer(query)]
    entities = []
    for match in _ENTITY_PHRASE.finditer(query):
        phrase = match.group(0)
        first_token = phrase.split()[0]
        if match.start() == 0 and first_token.casefold() in _QUESTION_WORDS:
            continue
        entities.append(phrase.casefold())
    entities = list(dict.fromkeys(entities))
    nested_relation = bool(_NESTED_RELATION.search(query))

    multi_hop = nested_relation or len(relation_matches) >= 2 or (
        len(entities) >= 2 and bool(conjunctions) and bool(relation_matches)
    )
    multi_concept = bool(conjunctions or comparisons or len(entities) >= 2)
    if multi_hop:
        label = "multi_hop"
    elif multi_concept:
        label = "multi_concept"
    else:
        label = "single_fact"

    reasons = []
    if nested_relation:
        reasons.append("nested_relation_pattern")
    if len(relation_matches) >= 2:
        reasons.append("multiple_relation_cues")
    if len(entities) >= 2:
        reasons.append("multiple_named_entities")
    if conjunctions:
        reasons.append("conjunction_present")
    if comparisons:
        reasons.append("comparison_present")
    if not reasons:
        reasons.append("single_target_question_pattern")

    return {
        "complexity": label,
        "features": {
            "entity_mentions": entities,
            "entity_count": len(entities),
            "conjunctions": conjunctions,
            "conjunction_count": len(conjunctions),
            "comparison_terms": comparisons,
            "comparison_count": len(comparisons),
            "relation_cues": relation_matches,
            "relation_count": len(relation_matches),
            "nested_relation_pattern": nested_relation,
            "reasons": reasons,
        },
    }
