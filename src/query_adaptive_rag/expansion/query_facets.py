"""Deterministic, conservative query and fact parsing for experimental M4-v3.

Only query/chunk text and ordinary chunk metadata are inspected here. This
module deliberately prefers missed facts over inferred or unsupported links.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any, Iterable, Mapping

_NAME_WORD = r"(?:[A-Z][A-Za-z0-9&'.’_-]*|[A-Z]{2,}|[A-Z]\.)"
_NAME = re.compile(rf"(?<![\w]){_NAME_WORD}(?:\s+{_NAME_WORD})*")
_STOP = {"a", "an", "and", "are", "as", "at", "before", "both", "but", "by", "during", "for", "from", "he", "her", "his", "how", "i", "in", "is", "it", "its", "me", "of", "on", "or", "she", "that", "the", "their", "they", "this", "to", "was", "we", "were", "when", "where", "what", "which", "who", "whom", "whose", "why", "you"}
_TYPE_WORDS = {"town", "city", "county", "state", "country", "film", "movie", "song", "company", "retailer", "river", "band", "album", "book", "organization", "person", "university", "school", "church"}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_LOCATION_TITLE_SUFFIXES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee",
    "texas", "utah", "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming",
    "district of columbia",
}


def normalize_entity(value: str) -> str:
    value = html.unescape(value).strip(" \t\r\n\"'“”()[]{}.,;:!?-")
    value = re.sub(r"(?i)(?:'s|’s)$", "", value)
    return re.sub(r"\s+", " ", value).casefold()


def extract_entities(text: str, *, allow_initial_single: bool = False) -> list[str]:
    """Return title-case name spans; a lone sentence-initial word is uncertain."""
    result: list[str] = []
    for match in _NAME.finditer(text):
        words = match.group().split()
        while words and normalize_entity(words[0]) in _STOP:
            words.pop(0)
        if not words:
            continue
        value = normalize_entity(" ".join(words))
        boundary = match.start() == 0 or text[max(0, match.start()-2):match.start()] in {". ", "? ", "! ", "\n"}
        if len(words) == 1 and boundary and not allow_initial_single:
            continue
        if value in _STOP or value in _TYPE_WORDS:
            continue
        if value not in result:
            result.append(value)
    return result


def document_title(chunk: Mapping[str, Any]) -> str | None:
    """Get normalized title metadata from a Day 1 SQuAD/Hotpot document ID."""
    doc_id = str(chunk.get("doc_id", ""))
    match = re.match(r"^(?:squad|hotpot):(.+):[0-9a-f]{8,}$", doc_id, re.I)
    return html.unescape(match.group(1)).strip() if match else None


def _document_title_base_alias(chunk: Mapping[str, Any], sentence: str) -> tuple[str, int] | None:
    """Resolve a location-disambiguated title only at a text-supported subject.

    The title is metadata, not evidence: the alias must be an exact phrase in
    the chunk and must begin this sentence's subject phrase (optionally after
    an article). The returned offset is the end of that phrase.
    """
    title = document_title(chunk)
    if not title or "," not in title:
        return None
    base, suffix = (part.strip() for part in title.rsplit(",", 1))
    if not base or normalize_entity(suffix) not in _LOCATION_TITLE_SUFFIXES:
        return None
    escaped = re.escape(base)
    exact_in_chunk = re.search(rf"(?<!\w){escaped}(?!\w)", str(chunk.get("text", "")), re.IGNORECASE)
    if not exact_in_chunk:
        return None
    subject_start = re.match(rf"\s*(?:(?:the|a|an)\s+)?({escaped})(?=$|[\s,;:])", sentence, re.IGNORECASE)
    if not subject_start:
        return None
    return normalize_entity(base), subject_start.end(1)


def extract_chunk_entities(chunk: Mapping[str, Any]) -> list[str]:
    """Include a title alias only when the title is also supported by text."""
    text = str(chunk.get("text", ""))
    entities = extract_entities(text)
    title = document_title(chunk)
    has_local_it_relation = bool(re.match(r"\s*it\b", text, re.I) and re.search(r"\b(?:is|was|near|along|located|based)\b", text, re.I))
    if title and (normalize_entity(title) in {normalize_entity(item) for item in entities} or has_local_it_relation):
        normalized = normalize_entity(title)
        if normalized not in entities:
            entities.append(normalized)
    return entities


@dataclass(frozen=True)
class RelationRequest:
    relation: str
    constrained_entity: str | None = None
    role: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"relation": self.relation, "constrained_entity": self.constrained_entity, "role": self.role}


@dataclass(frozen=True)
class QueryFrame:
    entities: tuple[str, ...]
    relations: tuple[RelationRequest, ...]
    answer_slot: str
    answer_type: str
    constraints: tuple[str, ...]
    cues: tuple[str, ...]
    comparison_entities: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"entities": list(self.entities), "relations": [r.as_dict() for r in self.relations],
                "answer_slot": self.answer_slot, "answer_type": self.answer_type,
                "constraints": list(self.constraints), "cues": list(self.cues),
                "comparison_entities": list(self.comparison_entities)}


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    source_chunk: str
    source_text: str
    span: tuple[int, int] = (0, 0)

    def key(self) -> tuple[str, str, str]:
        return normalize_entity(self.subject), self.relation, normalize_entity(self.object)

    def as_dict(self) -> dict[str, Any]:
        return {"subject": self.subject, "relation": self.relation, "object": self.object,
                "source_chunk": self.source_chunk, "source_text": self.source_text, "span": list(self.span)}


def parse_query_frame(query: str) -> QueryFrame:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    entities = tuple(extract_entities(query))
    q = query.casefold()
    requests: list[RelationRequest] = []
    cues: list[str] = []
    constraints: list[str] = []
    slot, answer_type = "unknown", "unknown"
    comparison: tuple[str, ...] = ()
    patterns = [
        (r"\bhometown\b|\bhome\s+town\b", "home_location", "place", "hometown"),
        (r"\b(?:when|what\s+(?:year|date))\b.{0,80}\bfounded\b|\bfounded\b.{0,60}\bwhen\b", "founded_on", "date", "founding date"),
        (r"\b(?:when|what\s+(?:year|date))\b.{0,80}\bborn\b|\bborn\s+on\b", "born_on", "date", "birth date"),
        (r"\bowned\s+by\b|\bwho\s+owns\b|\bowner\s+of\b", "owned_by", "person_or_organization", "ownership"),
        (r"\bbased\s+in\b|\bheadquartered\s+in\b|\blocated\s+in\b", "located_in", "place", "location"),
        (r"\bnear\b|\balong\b", "geographic_relation", "place", "near/along"),
        (r"\bwho\s+wrote\b|\bwritten\s+by\b|\bauthor(?:ed)?\s+of\b", "authored_by", "person", "authorship"),
        (r"\bwhat\s+company\s+(?:produced|made)\b|\bproduced\s+by\b", "produced_by", "organization", "production"),
        (r"\bwho\s+directed\b|\bdirected\s+by\b", "directed_by", "person", "direction"),
        (r"\bstars?\b|\bstarring\b|\bcast\b", "cast_member", "person", "cast"),
    ]
    # Comparison asks for a property of one of two named entities.
    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?),?\s+which\b", query, re.I)
    if between:
        comparison = tuple(extract_entities(f"{between.group(1)} {between.group(2)}")[:2])
        prop = re.search(r"\b(more|most|fewer|less|larger|farther|further|older|younger|longer|shorter)\s+([\w -]+?)(?:\?|$)", query, re.I)
        if prop:
            constraints.append(normalize_entity(prop.group(2)))
            requests.append(RelationRequest("comparison_property", None, normalize_entity(prop.group(2))))
            cues.extend(["between", prop.group(0).strip()])
            slot, answer_type = "comparison_choice", "one_of_compared_entities"
    if not requests:
        for pattern, relation, typ, cue in patterns:
            match = re.search(pattern, q)
            if not match:
                continue
            cues.append(match.group())
            constrained = None
            if relation in {"owned_by", "located_in", "authored_by", "produced_by", "directed_by", "cast_member"}:
                trailing = extract_entities(query[match.end():])
                constrained = trailing[0] if trailing else None
            if relation == "authored_by" and entities:
                constrained = entities[0]
                role = "subject"
            elif relation in {"directed_by", "cast_member"}:
                role = "object"
            else:
                role = None
            requests.append(RelationRequest(relation, constrained, role))
            slot, answer_type = relation, typ
        if "directed_by" in {r.relation for r in requests} and "cast_member" in {r.relation for r in requests}:
            slot, answer_type = "work_satisfying_relations", "film_or_movie"
        if answer_type == "unknown":
            typed = re.search(r"\bwhat\s+(town|city|county|film|movie|company|group|church|line)\b", q)
            if typed:
                slot = typed.group(1)
                answer_type = {"town":"town", "city":"city", "county":"county", "film":"film_or_movie", "movie":"film_or_movie", "company":"organization", "group":"group", "church":"church", "line":"product_line"}[slot]
    if re.search(r"\b(?:and|versus|vs\.?|between)\b", q):
        constraints.extend(e for e in entities if e not in constraints)
    return QueryFrame(entities, tuple(requests), slot, answer_type, tuple(dict.fromkeys(constraints)),
                      tuple(dict.fromkeys(cues)), comparison)


def _entities_with_spans(text: str) -> list[tuple[str, int, int]]:
    rows = []
    for m in _NAME.finditer(text):
        words = m.group().split()
        while words and normalize_entity(words[0]) in _STOP:
            words.pop(0)
        if not words:
            continue
        name = normalize_entity(" ".join(words))
        if name not in _STOP and name not in _TYPE_WORDS:
            rows.append((name, m.start(), m.end()))
    # Quoted titles can be names even when title casing varies.
    for m in re.finditer(r'["“]([^"”]{2,80})["”]', text):
        name = normalize_entity(m.group(1))
        if name and all(row[0] != name for row in rows):
            rows.append((name, m.start(1), m.end(1)))
    return sorted(rows, key=lambda row: row[1])


def _fact(subj: str, rel: str, obj: str, chunk: Mapping[str, Any], sent: str, start: int, end: int) -> Fact | None:
    subj, obj = normalize_entity(subj), normalize_entity(obj)
    if not subj or not obj or subj == obj:
        return None
    return Fact(subj, rel, obj, str(chunk.get("chunk_id", "")), sent.strip(), (start, end))


def _sentence_facts(sentence: str, chunk: Mapping[str, Any], offset: int) -> list[Fact]:
    ents = _entities_with_spans(sentence)
    facts: list[Fact] = []
    if not ents:
        return facts
    pred_re = re.compile(r"\bowned\s+by\b|\bbased\s+in\b|\bheadquartered\s+in\b|\blocated\s+in\b|\bnear\b|\balong\b|\bwritten\s+by\b|\bauthored\s+by\b|\bdirected\s+by\b|\bproduced\s+by\b|\bstarring\b|\bstars\b|\bwrote\b|\bdirected\b|\bproduced\b|\bowns\b|\bcomes\s+from\b|\bborn\s+in\b|\bfrom\b", re.I)
    predicates = list(pred_re.finditer(sentence))
    initial_subjects = [e for e in ents if e[2] <= (predicates[0].start() if predicates else len(sentence))]
    clause_subject = initial_subjects[0][0] if initial_subjects else None
    title_subject = _document_title_base_alias(chunk, sentence)
    for index, m in enumerate(predicates):
        left = [e for e in ents if e[2] <= m.start()]
        right = [e for e in ents if e[1] >= m.end()]
        if not left or not right:
            continue
        before, after = left[-1][0], right[0][0]
        if title_subject and title_subject[1] <= m.start():
            before = title_subject[0]
        cue = m.group().casefold()
        relation = ("owned_by" if cue in {"owned by", "owns"} else
                    "located_in" if cue in {"based in", "headquartered in", "located in"} else
                    "home_location" if cue in {"from", "comes from", "born in"} else
                    "near" if cue in {"near", "along"} else
                    "authored_by" if cue in {"written by", "authored by", "wrote"} else
                    "directed_by" if cue in {"directed by", "directed"} else
                    "produced_by" if cue in {"produced by", "produced"} else "cast_member")
        reverse = cue in {"wrote", "directed", "produced"}
        if index > 0 and not reverse:
            # Coordinated predicates (“Film was directed by A and stars B”)
            # retain the sentence's explicit subject instead of binding A.
            between = sentence[predicates[index - 1].end():m.start()]
            if clause_subject and re.search(r"\band\b|\bbut\b|\balso\b", between, re.I):
                before = clause_subject
        subj, obj = (after, before) if reverse else (before, after)
        value = _fact(subj, relation, obj, chunk, sentence, offset + m.start(), offset + m.end())
        if value:
            facts.append(value)
    # Entity-kind facts: “Gold Center was a mining town”.
    type_re = re.compile(r"\b(?:is|was|are|were)\s+(?:an?\s+|the\s+)?(?:[\w-]+\s+){0,3}(town|city|county|state|country|film|movie|song|company|retailer|river|band|album|book|organization|university|school|church)\b", re.I)
    for m in type_re.finditer(sentence):
        left = [e for e in ents if e[2] <= m.start()]
        if left:
            subject = title_subject[0] if title_subject and title_subject[1] <= m.start() else left[-1][0]
            value = _fact(subject, "entity_type", m.group(1).casefold(), chunk, sentence, offset + m.start(), offset + m.end())
            if value:
                facts.append(value)
    # Deterministic dates.
    date_re = re.compile(r"\b(?P<sub>[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*)\s+(?:was\s+)?(?P<rel>founded|born)\s+(?:on\s+)?(?P<obj>(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4})\b", re.I)
    for m in date_re.finditer(sentence):
        value = _fact(m.group("sub"), f"{m.group('rel').casefold()}_on", m.group("obj"), chunk, sentence, offset + m.start(), offset + m.end())
        if value:
            facts.append(value)
    # Filmography is a director fact only when the sentence explicitly says so.
    filmography = re.search(r"\b(?:his|her)\s+films?\s+include\b|\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*(?:'s|’s)?\s+films?\s+include\b", sentence, re.I)
    if filmography:
        antecedent = [e[0] for e in ents if e[2] <= filmography.start()]
        for title in re.finditer(r'["“]([^"”]+)["”]', sentence[filmography.end():]):
            if antecedent:
                value = _fact(title.group(1), "directed_by", antecedent[-1], chunk, sentence, offset + filmography.end() + title.start(), offset + filmography.end() + title.end())
                if value:
                    facts.append(value)
    director_list = re.search(r"\bas\s+a\s+director,?\s+([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*)\s+listed\s+([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*)\s+among\s+(?:his|her)\s+films?", sentence, re.I)
    if director_list:
        value = _fact(director_list.group(2), "directed_by", director_list.group(1), chunk, sentence,
                      offset + director_list.start(), offset + director_list.end())
        if value:
            facts.append(value)
    unique = {}
    for fact in facts:
        unique.setdefault(fact.key(), fact)
    return list(unique.values())


def extract_facts(chunks: Iterable[Mapping[str, Any]]) -> list[Fact]:
    """Extract explicit relation tuples; resolve only a clear local pronoun.

    A pronoun in an adjacent sentence/chunk is substituted only if the prior
    sentence explicitly named exactly one subject. Title metadata can resolve
    a local “it” but cannot create a fact without an explicit predicate.
    Ambiguous or cross-document pronouns stay unresolved.
    """
    ordered = sorted(chunks, key=lambda c: (str(c.get("doc_id", "")), int(c.get("start", 0)), str(c.get("chunk_id", ""))))
    facts: list[Fact] = []
    last_subject: dict[str, str] = {}
    last_chunk: dict[str, str] = {}
    for chunk in ordered:
        doc = str(chunk.get("doc_id", "")); cid = str(chunk.get("chunk_id", "")); offset = 0
        adjacent = last_chunk.get(doc) == chunk.get("prev_chunk_id")
        previous_subject = last_subject.get(doc) if adjacent else None
        for sentence in _SENTENCE_SPLIT.split(str(chunk.get("text", ""))):
            if not sentence.strip():
                offset += len(sentence) + 1
                continue
            pronoun = re.search(r"\b(it|he|she|they|his|her|its|their)\b", sentence, re.I)
            antecedent = previous_subject
            # A title may resolve a local “it” only when the same chunk states
            # an explicit relation. The title itself never creates a fact.
            if not antecedent and re.match(r"\s*it\b", sentence, re.I):
                antecedent = normalize_entity(document_title(chunk) or "") or None
            resolved = sentence
            # Resolve only if antecedent is explicit and this sentence has no
            # competing proper name. It is local, conservative, and logged in facts.
            names = extract_entities(sentence)
            explicit_before_pronoun = any(m.start() < pronoun.start() for m in _NAME.finditer(sentence)) if pronoun else False
            if pronoun and antecedent and not explicit_before_pronoun:
                replacement = " ".join(part.capitalize() for part in antecedent.split())
                if pronoun.group().casefold() in {"his", "her", "its", "their"}:
                    replacement += "'s"
                resolved = sentence[:pronoun.start()] + replacement + sentence[pronoun.end():]
            sentence_facts = _sentence_facts(resolved, chunk, offset)
            facts.extend(sentence_facts)
            if sentence_facts:
                subjects = {fact.subject for fact in sentence_facts}
                previous_subject = next(iter(subjects)) if len(subjects) == 1 else None
            else:
                names = extract_entities(sentence)
                if len(names) == 1 and sentence.lstrip().casefold().startswith(names[0]):
                    previous_subject = names[0]
                else:
                    previous_subject = None
            offset += len(sentence) + 1
        if previous_subject:
            last_subject[doc] = previous_subject
        else:
            last_subject.pop(doc, None)
        last_chunk[doc] = cid
    unique = {}
    for fact in facts:
        unique.setdefault((fact.key(), fact.source_chunk), fact)
    return list(unique.values())


def fact_matches_request(fact: Fact, request: RelationRequest) -> bool:
    if fact.relation != request.relation:
        return False
    if not request.constrained_entity:
        return True
    entity = normalize_entity(request.constrained_entity)
    return entity in {normalize_entity(fact.subject), normalize_entity(fact.object)}


def fact_is_answer_type(fact: Fact, answer_type: str) -> bool:
    allowed = {"town": {"town"}, "city": {"city"}, "county": {"county"},
               "place": {"town", "city", "county", "state", "country", "river"},
               "film_or_movie": {"film", "movie"}, "organization": {"company", "retailer", "organization"},
               "group": {"band", "group"}, "church": {"church"}}
    return fact.relation == "entity_type" and normalize_entity(fact.object) in allowed.get(answer_type, set())
