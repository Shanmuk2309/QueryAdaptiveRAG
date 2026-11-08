"""M4-v3: query-facet-aware, incremental semantic neighbor expansion.

This module is isolated from M3, M4, and M4-v2. Runtime selection uses only
the query, retrieved context, and cached chunk embeddings; evaluation labels
are not accepted by this API.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from ..chunking.text_utils import OffsetTokenizer
from ..retrieval.dense import SearchResponse
from .context_expander import ExpansionRule, TokenBudgetError, assemble_context
from .query_facets import Fact, QueryFrame, extract_facts, parse_query_frame


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "was",
    "were", "what", "which", "who", "whom", "whose", "with", "but", "that",
}
_RELATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "ownership": (r"\bowned\s+by\b", r"\bowns?\b", r"\bowner\b"),
    "location": (r"\bbased\s+in\b", r"\blocated\s+in\b", r"\bheadquartered\s+in\b"),
    "director": (r"\bdirected\s+by\b", r"\bdirector\b", r"\bdirected\b"),
    "cast": (r"\bstars?\b", r"\bstarring\b", r"\bcast\b"),
    "authorship": (r"\bwrote\b", r"\bwritten\s+by\b", r"\bauthor\b", r"\bsongwriter\b"),
    "creation": (r"\bcreated\s+by\b", r"\bco-created\b", r"\bcreator\b"),
    "featured_in": (r"\bfeatured\s+(?:characters\s+)?(?:in|from)\b", r"\bfeatures?\b"),
}
_ENTITY_RE = re.compile(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*")
_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")


@dataclass(frozen=True)
class QueryFacets:
    entities: tuple[str, ...]
    relations: tuple[str, ...]
    keywords: tuple[str, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple([*(f"entity:{x}" for x in self.entities),
                      *(f"relation:{x}" for x in self.relations),
                      *(f"keyword:{x}" for x in self.keywords)])


@dataclass(frozen=True)
class ExpansionV3Config:
    min_query_similarity: float = 0.20
    min_incremental_gain: float = 0.01
    corroboration_similarity: float = 0.35
    near_duplicate_3gram_containment: float = 0.85
    semantic_redundancy_warning: float = 0.85
    max_corroborations_per_facet: int = 1
    entity_weight: float = 2.0
    relation_weight: float = 2.0
    keyword_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("min_query_similarity", "corroboration_similarity",
                     "near_duplicate_3gram_containment", "semantic_redundancy_warning"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_incremental_gain < 0:
            raise ValueError("min_incremental_gain must be non-negative")
        if self.max_corroborations_per_facet < 0:
            raise ValueError("max_corroborations_per_facet must be non-negative")
        if min(self.entity_weight, self.relation_weight, self.keyword_weight) < 0:
            raise ValueError("facet weights must be non-negative")


@dataclass(frozen=True)
class ExpansionV3Result:
    seed_chunk_ids: list[str]
    seed_ranks: list[int]
    expanded_chunk_ids: list[str]
    similarity_scores: dict[str, float]
    redundancy_scores: dict[str, float]
    context_text: str
    context_token_count: int
    token_budget: int
    latency_ms: float
    candidate_decisions: list[dict[str, Any]]


def extract_query_facets(query: str) -> QueryFacets:
    """Compatibility view of the richer fact-oriented parser."""
    frame = parse_query_frame(query)
    entities = frame.entities
    relations = tuple(dict.fromkeys(request.relation for request in frame.relations))
    keywords = tuple()
    return QueryFacets(entities, relations, keywords)


def _compatible_request_fact(fact: Fact, request: Any, frame: QueryFrame) -> bool:
    aliases = {"geographic_relation": {"near", "located_in"},
               "home_location": {"located_in"},
               "comparison_property": {"entity_type"}}
    if fact.relation not in aliases.get(request.relation, {request.relation}):
        return False
    if request.constrained_entity:
        return request.constrained_entity.casefold() in {fact.subject.casefold(), fact.object.casefold()}
    return any(entity in {fact.subject.casefold(), fact.object.casefold()} for entity in frame.entities)


def _incremental_fact_decision(frame: QueryFrame, selected_facts: list[Fact], candidate_facts: list[Fact]
                               ) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]], str]:
    """Accept only a new explicit fact binding an uncovered query relation/slot."""
    selected_keys = {fact.key() for fact in selected_facts}
    required = list(frame.relations)

    def tied_to_same_answer(fact: Fact, request: Any, pool: list[Fact]) -> bool:
        if frame.answer_slot != "work_satisfying_relations" or len(required) < 2:
            return True
        other_requests = [item for item in required if item.relation != request.relation]
        return any(other.subject == fact.subject and other.key() != fact.key()
                   and any(_compatible_request_fact(other, item, frame) for item in other_requests)
                   for other in pool)

    selected_bindings = [
        (request, fact) for request in frame.relations for fact in selected_facts
        if _compatible_request_fact(fact, request, frame) and tied_to_same_answer(fact, request, selected_facts)
    ]
    uncovered = [request for request in frame.relations
                 if not any(req.relation == request.relation for req, _ in selected_bindings)]
    supported: list[dict[str, Any]] = []
    for fact in candidate_facts:
        if fact.key() in selected_keys:
            continue
        for request in uncovered:
            if (_compatible_request_fact(fact, request, frame)
                    and tied_to_same_answer(fact, request, selected_facts + candidate_facts)):
                supported.append({"fact": fact.as_dict(), "query_relation": request.as_dict(), "link": "direct_query_relation"})
                break
        if supported and supported[-1].get("fact", {}).get("source_chunk") == fact.source_chunk:
            continue
        # Answer-type facts are useful only when their subject is explicitly
        # connected by a selected fact to a query entity/relation.
        if fact.relation == "entity_type" and frame.answer_type != "unknown":
            expected = {"town": {"town"}, "place": {"town", "city", "county", "state", "country", "river"},
                        "city": {"city"}, "county": {"county"}, "film_or_movie": {"film", "movie"},
                        "organization": {"company", "retailer", "organization"}, "group": {"band", "group"}}
            if fact.object not in expected.get(frame.answer_type, set()):
                continue
            connected = [old for old in selected_facts if fact.subject in {old.subject, old.object}
                         and any(_compatible_request_fact(old, request, frame) for request in frame.relations)]
            if connected:
                supported.append({"fact": fact.as_dict(), "query_relation": None, "link": "selected_fact_chain",
                                  "chain": [old.as_dict() for old in connected]})
    if supported:
        return True, supported, [f.as_dict() for f in candidate_facts], "adds_new_query_fact"
    if not candidate_facts:
        return False, [], [], "no_supported_query_fact"
    if all(f.key() in selected_keys for f in candidate_facts):
        return False, [], [f.as_dict() for f in candidate_facts], "fact_already_established"
    return False, [], [f.as_dict() for f in candidate_facts], "no_supported_query_fact"


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.casefold())


def _contains_phrase(text: str, phrase: str) -> bool:
    haystack = " ".join(_words(text))
    needle = " ".join(_words(phrase))
    return bool(needle) and f" {needle} " in f" {haystack} "


def _entity_supported(entity: str, text: str) -> bool:
    if _contains_phrase(text, entity):
        return True
    # Allow a salient surname when the query entity is a multiword name.
    parts = entity.split()
    return len(parts) > 1 and _contains_phrase(text, parts[-1])


def _relation_supported(relation: str, text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE)
               for pattern in _RELATION_PATTERNS[relation])


def _entity_relation_supported(facets: QueryFacets, text: str) -> list[str]:
    """Return relation facets locally tied to at least one query entity."""
    matches: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        has_entity = any(_entity_supported(entity, sentence) for entity in facets.entities)
        if not has_entity:
            continue
        for relation in facets.relations:
            if _relation_supported(relation, sentence) and relation not in matches:
                matches.append(relation)
    return matches


def _facet_support(facets: QueryFacets, text: str, config: ExpansionV3Config) -> set[str]:
    supported = {f"entity:{entity}" for entity in facets.entities if _entity_supported(entity, text)}
    supported.update(f"relation:{relation}" for relation in _entity_relation_supported(facets, text))
    text_words = set(_words(text))
    supported.update(f"keyword:{term}" for term in facets.keywords if term in text_words)
    return supported


def _facet_weight(facet: str, config: ExpansionV3Config) -> float:
    return {"entity": config.entity_weight, "relation": config.relation_weight,
            "keyword": config.keyword_weight}[facet.split(":", 1)[0]]


def _incremental_gain(facets: QueryFacets, selected_text: str, candidate_text: str,
                      config: ExpansionV3Config) -> tuple[float, list[str], list[str]]:
    selected_support = _facet_support(facets, selected_text, config)
    candidate_support = _facet_support(facets, candidate_text, config)
    additions = sorted(facet for facet in candidate_support - selected_support
                       if _facet_weight(facet, config) > 0)
    denominator = sum(_facet_weight(facet, config) for facet in facets.labels)
    gain = (sum(_facet_weight(facet, config) for facet in additions) / denominator) if denominator else 0.0
    return gain, additions, sorted(selected_support)


def _normalized_text(text: str) -> str:
    return " ".join(_words(text))


def _threegrams(text: str) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))}


def _max_3gram_containment(candidate_text: str, selected: list[dict[str, Any]]) -> float:
    candidate_grams = _threegrams(candidate_text)
    if not candidate_grams:
        candidate_words = _words(candidate_text)
        selected_words = set(_words(" ".join(chunk["text"] for chunk in selected)))
        return 1.0 if candidate_words and all(word in selected_words for word in candidate_words) else 0.0
    selected_grams: set[tuple[str, ...]] = set()
    for chunk in selected:
        selected_grams.update(_threegrams(chunk["text"]))
    return len(candidate_grams & selected_grams) / len(candidate_grams)


def _unit(vector: Any, chunk_id: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm == 0.0:
        raise ValueError(f"invalid embedding for {chunk_id!r}")
    return value / norm


def _neighbor_ids(chunk: Mapping[str, Any]) -> list[str]:
    return [item for item in (chunk.get("prev_chunk_id"), chunk.get("next_chunk_id")) if item]


def _valid_neighbor(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    if current.get("doc_id") != candidate.get("doc_id") or current.get("chunk_id") == candidate.get("chunk_id"):
        return False
    current_id = current["chunk_id"]
    return candidate.get("prev_chunk_id") == current_id or candidate.get("next_chunk_id") == current_id


def expand_context_v3(
    search: SearchResponse,
    query: str,
    complexity: str,
    chunks: list[dict[str, Any]],
    chunk_embeddings: Mapping[str, np.ndarray],
    tokenizer: OffsetTokenizer,
    score_chunk: Callable[[np.ndarray, str], float],
    rules: Mapping[str, ExpansionRule],
    token_budget: int,
    config: ExpansionV3Config | None = None,
) -> ExpansionV3Result:
    """Expand unchanged M3 seeds using relevance, incremental facets, and redundancy."""
    started = perf_counter()
    config = config or ExpansionV3Config()
    if complexity not in rules:
        raise ValueError(f"No expansion rule configured for {complexity!r}")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if not search.hits:
        raise ValueError("at least one M3 seed chunk is required")
    if query.strip() == "":
        raise ValueError("query must be non-empty")
    rule = rules[complexity]
    if rule.max_depth < 0 or rule.max_expansions < 0:
        raise ValueError("expansion depth and count must be non-negative")

    store = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(store) != len(chunks):
        raise ValueError("semantic chunk IDs must be unique")
    seeds = [hit.chunk for hit in search.hits]
    seed_ids = [hit.chunk["chunk_id"] for hit in search.hits]
    seed_ranks = [hit.rank for hit in search.hits]
    for chunk_id in seed_ids:
        if chunk_id not in store or chunk_id not in chunk_embeddings:
            raise ValueError(f"M3 seed {chunk_id!r} is missing from chunk store or embedding map")
    context, token_count = assemble_context(seeds, tokenizer)
    if token_count > token_budget:
        raise TokenBudgetError(f"M3 seed context requires {token_count} tokens; budget is {token_budget}")

    frame = parse_query_frame(query)
    selected = list(seeds)
    selected_ids = set(seed_ids)
    selected_vectors = {cid: _unit(chunk_embeddings[cid], cid) for cid in seed_ids}
    expanded_ids: list[str] = []
    similarity_scores = {hit.chunk["chunk_id"]: float(hit.score) for hit in search.hits}
    redundancy_scores: dict[str, float] = {}
    decisions: list[dict[str, Any]] = []
    visited = set(seed_ids)
    frontier = list(seeds)

    for depth in range(1, rule.max_depth + 1):
        candidates: dict[str, dict[str, Any]] = {}
        for current in frontier:
            for candidate_id in _neighbor_ids(current):
                candidate = store.get(candidate_id)
                if candidate is None or candidate_id in visited:
                    continue
                visited.add(candidate_id)
                if _valid_neighbor(current, candidate):
                    candidates[candidate_id] = candidate
                else:
                    decisions.append({"candidate_chunk_id": candidate_id, "depth": depth,
                                      "accepted": False, "reason": "invalid_neighbor_link"})

        ranked: list[tuple[float, str, dict[str, Any], np.ndarray]] = []
        for candidate_id, candidate in candidates.items():
            query_score = float(score_chunk(search.query_embedding, candidate_id))
            similarity_scores[candidate_id] = query_score
            vector_raw = chunk_embeddings.get(candidate_id)
            if vector_raw is None:
                raise ValueError(f"candidate embedding missing for {candidate_id!r}")
            candidate_vector = _unit(vector_raw, candidate_id)
            if candidate_vector.shape != next(iter(selected_vectors.values())).shape:
                raise ValueError(f"embedding dimension mismatch for candidate chunk {candidate_id!r}")
            ranked.append((query_score, candidate_id, candidate, candidate_vector))

        ranked.sort(key=lambda item: (-item[0], item[2]["doc_id"], item[2]["start"], item[1]))
        next_frontier: list[dict[str, Any]] = []
        for query_score, candidate_id, candidate, candidate_vector in ranked:
            selected_text, _ = assemble_context(selected, tokenizer)
            candidate_text = str(candidate["text"])
            context_facts = extract_facts(selected + [candidate])
            selected_facts = [fact for fact in context_facts if fact.source_chunk in selected_ids]
            candidate_facts = [fact for fact in context_facts if fact.source_chunk == candidate_id]
            fact_accept, supporting_facts, candidate_fact_log, fact_reason = _incremental_fact_decision(
                frame, selected_facts, candidate_facts
            )
            normalized_candidate = _normalized_text(candidate_text)
            exact_duplicate = bool(normalized_candidate) and normalized_candidate in {
                _normalized_text(item["text"]) for item in selected if _normalized_text(item["text"])
            }
            lexical_overlap = _max_3gram_containment(candidate_text, selected)
            semantic_redundancy = max(
                float(np.dot(candidate_vector, selected_vector)) for selected_vector in selected_vectors.values()
            )
            redundancy_scores[candidate_id] = semantic_redundancy
            decision: dict[str, Any] = {
                "candidate_chunk_id": candidate_id, "depth": depth,
                "query_similarity": query_score,
                "query_frame": frame.as_dict(),
                "selected_facts": [fact.as_dict() for fact in selected_facts],
                "candidate_facts": candidate_fact_log,
                "supporting_fact_chain": supporting_facts,
                "max_3gram_containment": lexical_overlap,
                "semantic_redundancy": semantic_redundancy,
                "semantic_redundancy_warning": semantic_redundancy >= config.semantic_redundancy_warning,
                "accepted": False, "reason": "",
            }
            if query_score < config.min_query_similarity:
                decision["reason"] = "query_relevance_below_floor"
            elif exact_duplicate:
                decision["reason"] = "exact_duplicate"
            elif fact_accept:
                decision["accepted"] = True
                decision["reason"] = fact_reason
            elif lexical_overlap >= config.near_duplicate_3gram_containment:
                decision["reason"] = "near_duplicate_without_new_fact"
            else:
                decision["reason"] = fact_reason
            if len(expanded_ids) >= rule.max_expansions:
                if decision["accepted"]:
                    decision["accepted"] = False
                    decision["reason"] = "expansion_limit_reached"
                decisions.append(decision)
                continue
            if not decision["accepted"]:
                decisions.append(decision)
                continue
            if candidate_id in selected_ids:
                decision["accepted"] = False
                decision["reason"] = "already_selected"
                decisions.append(decision)
                continue
            proposed_text, proposed_count = assemble_context(selected + [candidate], tokenizer)
            if proposed_text == context:
                decision["reason"] = "deduplicated_text"
                decision["accepted"] = False
                decisions.append(decision)
                continue
            if proposed_count > token_budget:
                decision["reason"] = "token_budget_exceeded"
                decision["token_count_if_added"] = proposed_count
                decision["token_budget"] = token_budget
                decision["accepted"] = False
                decisions.append(decision)
                continue
            selected.append(candidate)
            selected_ids.add(candidate_id)
            selected_vectors[candidate_id] = candidate_vector
            expanded_ids.append(candidate_id)
            context, token_count = proposed_text, proposed_count
            decision["token_count_if_added"] = proposed_count
            decisions.append(decision)
            next_frontier.append(candidate)
        frontier = next_frontier
        if not frontier or len(expanded_ids) >= rule.max_expansions:
            break

    return ExpansionV3Result(
        seed_chunk_ids=seed_ids, seed_ranks=seed_ranks, expanded_chunk_ids=expanded_ids,
        similarity_scores=similarity_scores, redundancy_scores=redundancy_scores,
        context_text=context, context_token_count=token_count, token_budget=token_budget,
        latency_ms=(perf_counter() - started) * 1000.0, candidate_decisions=decisions,
    )
