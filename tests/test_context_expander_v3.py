import inspect
import re

import numpy as np
import pytest

from query_adaptive_rag.expansion.context_expander import ExpansionRule
from query_adaptive_rag.expansion.context_expander_v3 import ExpansionV3Config, expand_context_v3
from query_adaptive_rag.expansion.query_facets import extract_chunk_entities, extract_entities, extract_facts, parse_query_frame
from query_adaptive_rag.retrieval.dense import DenseRetriever, SearchHit, SearchResponse


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, verbose=True):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return result


def chunk(cid, text, start, prev=None, next_=None, doc="doc"):
    return {"chunk_id": cid, "doc_id": doc, "text": text, "start": start, "end": start + len(text),
            "prev_chunk_id": prev, "next_chunk_id": next_, "token_count": len(text.split()), "method": "semantic"}


def run_case(question, seed_text, candidates, *, score=0.6, token_budget=100, rule=None, doc="doc"):
    next_id = candidates[0][0] if candidates else None
    seed = chunk("seed", seed_text, 100, prev="c", next_=next_id, doc=doc)
    chunks = [seed]
    embeddings = {"seed": np.array([1.0, 0.0], dtype=np.float32)}
    scores = {}
    for cid, text, vector, start in candidates:
        chunks.append(chunk(cid, text, start, prev="seed", doc=doc))
        embeddings[cid] = np.asarray(vector, dtype=np.float32)
        scores[cid] = score if not isinstance(score, dict) else score[cid]
    search = SearchResponse([SearchHit(seed, 0.9, 1)], np.array([1.0, 0.0], dtype=np.float32), 0.2)
    result = expand_context_v3(search, question, "multi_hop", chunks, embeddings, TinyTokenizer(),
                               lambda _q, cid: scores[cid], {"multi_hop": rule or ExpansionRule(2, 4)},
                               token_budget, ExpansionV3Config())
    return result, search, chunks, embeddings


def run_multi_seed_case(question, seeds, candidate):
    """Run the expander with query-relevant and same-document neighbor seeds."""
    chunks = [*seeds, candidate]
    vectors = {item["chunk_id"]: np.array([1.0, 0.0], dtype=np.float32) for item in seeds}
    vectors[candidate["chunk_id"]] = np.array([0.8, 0.6], dtype=np.float32)
    hits = [SearchHit(seed, 0.9 - i * 0.1, i + 1) for i, seed in enumerate(seeds)]
    search = SearchResponse(hits, np.array([1.0, 0.0], dtype=np.float32), 0.2)
    result = expand_context_v3(search, question, "multi_hop", chunks, vectors, TinyTokenizer(),
                               lambda _q, cid: 0.22 if cid == candidate["chunk_id"] else 0.5,
                               {"multi_hop": ExpansionRule(1, 4)}, 3072, ExpansionV3Config())
    return result


def test_entity_extraction_filters_question_words_and_sentence_initial_common_words():
    assert not set(extract_entities("When was it founded during the war?")) & {"when", "during", "the", "what"}
    assert extract_entities("During winter, Harold Ramis wrote Arizona.") == ["harold ramis", "arizona"]


def test_proper_names_possessives_initials_and_document_title_are_normalized():
    assert "harold ramis" in extract_entities("Harold Ramis's films include Groundhog Day.")
    assert "j. r. r. tolkien" in extract_entities("J. R. R. Tolkien wrote the book.")
    meta = chunk("c", "Gold Center was a town.", 0, doc="squad:Gold Center:1234567890abcdef")
    assert "gold center" in extract_chunk_entities(meta)
    alias_chunk = chunk("c", "It was also near the Amargosa River.", 0,
                        doc="squad:Gold Center:1234567890abcdef")
    alias_facts = extract_facts([alias_chunk])
    assert ("gold center", "near", "amargosa river") in {fact.key() for fact in alias_facts}


@pytest.mark.parametrize("query,relation,slot", [
    ("What is Harold Ramis's hometown?", "home_location", "home_location"),
    ("When was La Senza founded?", "founded_on", "founded_on"),
    ("Who is La Senza owned by?", "owned_by", "owned_by"),
    ("What town is near the Amargosa River?", "geographic_relation", "geographic_relation"),
    ("Who wrote Arizona?", "authored_by", "authored_by"),
    ("What company produced the film?", "produced_by", "produced_by"),
    ("Between Gold Center and Sarcobatus Flat, which is farther north?", "comparison_property", "comparison_choice"),
])
def test_query_relation_and_answer_slot_patterns(query, relation, slot):
    frame = parse_query_frame(query)
    assert relation in {r.relation for r in frame.relations}
    assert frame.answer_slot == slot


def test_active_and_passive_relation_facts_are_normalized():
    chunks = [chunk("x", "Arizona was written by Kenny Young.", 0),
              chunk("y", "Kenny Young wrote Arizona.", 40)]
    facts = extract_facts(chunks)
    assert ("arizona", "authored_by", "kenny young") in {f.key() for f in facts}


def test_ambiguous_pronoun_is_not_resolved():
    chunks = [chunk("a", "Harold Ramis met Bill Murray.", 0),
              chunk("b", "He directed a film.", 50, prev="a")]
    assert not any(f.relation == "directed_by" for f in extract_facts(chunks))
    cross_doc = [chunk("a", "Harold Ramis was a director.", 0, doc="d1"),
                 chunk("b", "He directed Groundhog Day.", 0, prev="a", doc="d2")]
    assert not extract_facts(cross_doc)


def test_clear_local_pronoun_is_resolved_only_with_one_subject():
    chunks = [chunk("a", "Harold Ramis was a director.", 0),
              chunk("b", "He directed Groundhog Day.", 50, prev="a")]
    # The predicate is explicit and the only local named antecedent is Ramis.
    assert ("groundhog day", "directed_by", "harold ramis") in {f.key() for f in extract_facts(chunks)}


def test_gold_center_location_title_alias_completes_explicit_fact_chain():
    doc_id = "hotpot:Gold Center, Nevada:0cedd65e19ed"
    query = "What unincorporated town along the Amargosa River is near the Sarcobatus Flat?"
    selected = chunk("selected", "Gold Center, Nevada was also near the Amargosa River.", 100,
                     prev="candidate", next_="candidate", doc=doc_id)
    candidate = chunk("candidate", "Gold Center was a mining town.", 0,
                      prev="selected", doc=doc_id)
    facts = extract_facts([selected])
    assert ("gold center", "near", "amargosa river") in {fact.key() for fact in facts}

    result = run_multi_seed_case(query, [selected], candidate)
    assert result.expanded_chunk_ids == ["candidate"]
    decision = result.candidate_decisions[0]
    assert decision["reason"] == "adds_new_query_fact"
    assert any(item["fact"]["subject"] == "gold center" and
               item["fact"]["relation"] == "entity_type" and
               item["link"] == "selected_fact_chain"
               for item in decision["supporting_fact_chain"])


def test_title_alias_without_textual_subject_and_explicit_relation_creates_no_fact():
    title_only = chunk("c", "A historic settlement was discussed.", 0,
                       doc="hotpot:Gold Center, Nevada:0cedd65e19ed")
    title_without_leading_alias = chunk("c", "The area was near the Amargosa River.", 0,
                                        doc="hotpot:Gold Center, Nevada:0cedd65e19ed")
    assert extract_facts([title_only]) == []
    assert extract_facts([title_without_leading_alias]) == []
    result, *_ = run_case(
        "What town is near the Amargosa River?",
        "The Amargosa River is in Nevada.",
        [("candidate", "A historic settlement was discussed.", [0.8, 0.6], 0)],
        doc="hotpot:Gold Center, Nevada:0cedd65e19ed",
    )
    assert result.expanded_chunk_ids == []
    assert result.candidate_decisions[0]["reason"] == "no_supported_query_fact"


def test_gold_center_repeated_relation_is_not_accepted_as_corroboration():
    doc_id = "hotpot:Gold Center, Nevada:0cedd65e19ed"
    selected = chunk("selected", "Gold Center, Nevada was near the Amargosa River.", 100,
                     prev="candidate", next_="candidate", doc=doc_id)
    candidate = chunk("candidate", "Gold Center, Nevada was also near the Amargosa River.", 0,
                      prev="selected", doc=doc_id)
    result = run_multi_seed_case("What town is near the Amargosa River?", [selected], candidate)
    assert result.expanded_chunk_ids == []
    assert result.candidate_decisions[0]["accepted"] is False
    assert result.candidate_decisions[0]["reason"] in {
        "fact_already_established", "near_duplicate_without_new_fact"
    }


def test_harold_ramis_film_facts_do_not_infer_cast_and_repeats_rejected():
    query = "Which film was directed by Harold Ramis and stars Bill Murray?"
    seed_text = "Groundhog Day was directed by Harold Ramis and stars Bill Murray."
    facts = extract_facts([chunk("s", seed_text, 0)])
    assert ("groundhog day", "directed_by", "harold ramis") in {f.key() for f in facts}
    assert ("groundhog day", "cast_member", "bill murray") in {f.key() for f in facts}
    result, *_ = run_case(query, seed_text,
                          [("c", "As a director, Ramis listed Groundhog Day among his films.", [0.8, 0.6], 200)])
    assert result.expanded_chunk_ids == []
    assert result.candidate_decisions[0]["reason"] in {"fact_already_established", "no_supported_query_fact"}
    unrelated_cast, *_ = run_case(query,
        "Groundhog Day was directed by Harold Ramis.",
        [("c", "The Royal Tenenbaums stars Bill Murray.", [0.8, 0.6], 200)])
    assert unrelated_cast.expanded_chunk_ids == []


def test_lasenza_espresso_neighbor_is_rejected():
    la_senza = chunk("la", "La Senza is owned by L Brands. L Brands is based in Columbus, Ohio.", 0,
                     doc="hotpot:La Senza:409685300623")
    espresso_seed = chunk("espresso_seed", "An espresso machine uses hot water.", 100,
                          next_="espresso_candidate", doc="hotpot:Espresso machine:ece974276615")
    espresso_candidate = chunk("espresso_candidate",
        "An espresso machine brews coffee by forcing pressurized water near boiling point through a puck of ground coffee.",
        200, prev="espresso_seed", doc="hotpot:Espresso machine:ece974276615")
    result = run_multi_seed_case("The La Senza brand is owned by a retailer based in what Ohio city?",
                                 [la_senza, espresso_seed], espresso_candidate)
    assert result.expanded_chunk_ids == []
    candidate_facts = result.candidate_decisions[0]["candidate_facts"]
    assert not any(f["subject"] == "la senza" or f["object"] in {"la senza", "columbus", "ohio"}
                   or f["relation"] in {"owned_by", "located_in"} for f in candidate_facts)
    assert result.candidate_decisions[0]["reason"] == "no_supported_query_fact"


def test_arizona_unrelated_authorship_neighbor_is_rejected():
    arizona = chunk("arizona_seed", "Arizona was written by Kenny Young and recorded by Mark Lindsay.", 0,
                    doc="hotpot:Arizona (song):295225603fd5")
    norman_seed = chunk("norman_seed", "Jimmy Norman was a songwriter.", 100,
                        next_="norman_candidate", doc="hotpot:Jimmy Norman:7de26f3c2a12")
    norman_candidate = chunk("norman_candidate",
        "Jimmy Norman worked with The Coasters and later released a solo album. He received few royalties for his writing.",
        200, prev="norman_seed", doc="hotpot:Jimmy Norman:7de26f3c2a12")
    result = run_multi_seed_case("The song Arizona was recorded by Paul Revere and Mark Lindsay but who wrote it?",
                                 [arizona, norman_seed], norman_candidate)
    assert result.expanded_chunk_ids == []
    candidate_facts = result.candidate_decisions[0]["candidate_facts"]
    assert not any(f["subject"] == "arizona" and f["relation"] == "authored_by" and
                   f["object"] == "jimmy norman" for f in candidate_facts)
    assert not any("arizona" in {f["subject"], f["object"]} for f in candidate_facts)
    assert result.candidate_decisions[0]["reason"] == "no_supported_query_fact"


def test_generic_keyword_only_candidate_is_rejected():
    result, *_ = run_case("What town is near the Amargosa River?", "The Amargosa River is in Nevada.",
                          [("c", "The area includes a town and several small roads.", [0.8, 0.6], 200)])
    assert result.expanded_chunk_ids == []
    assert result.candidate_decisions[0]["candidate_facts"] == []


def test_already_covered_fact_rejected_without_corroboration_path():
    result, *_ = run_case("Who wrote Arizona?", "Arizona was written by Kenny Young.",
                          [("c", "Kenny Young wrote Arizona.", [0.8, 0.6], 200)])
    assert result.expanded_chunk_ids == []
    assert result.candidate_decisions[0]["accepted"] is False
    assert not any("corrobor" in row["reason"] for row in result.candidate_decisions)


def test_m3_seed_ids_order_ranks_embedding_and_scores_are_unchanged():
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.asarray([[1, 0] if text == "query" else [0.9, 0.1] for text in texts], dtype=np.float32)
    seed = chunk("seed", "Alex Smith acted in a film.", 100, prev="n", next_="n")
    neighbor = chunk("n", "Alex Smith was directed by Dana Lee.", 0, next_="seed")
    retriever = DenseRetriever(Encoder()); retriever.build([seed, neighbor])
    before = retriever.retrieve("query", 1)
    ids = [hit.chunk["chunk_id"] for hit in before.hits]; ranks = [hit.rank for hit in before.hits]
    scores = [hit.score for hit in before.hits]; emb = before.query_embedding.copy()
    vectors = {"seed": retriever.vectors[0], "n": retriever.vectors[1]}
    expanded = expand_context_v3(before, "Who directed Alex Smith?", "multi_hop", [seed, neighbor], vectors,
                                 TinyTokenizer(), retriever.score, {"multi_hop": ExpansionRule(1, 2)}, 100)
    assert expanded.seed_chunk_ids == ids and expanded.seed_ranks == ranks
    assert [h.chunk["chunk_id"] for h in retriever.retrieve("query", 1).hits] == ids
    assert [h.score for h in before.hits] == scores
    np.testing.assert_array_equal(before.query_embedding, emb)


def test_runtime_api_does_not_accept_evaluation_labels_or_gold_inputs():
    params = set(inspect.signature(expand_context_v3).parameters)
    assert not params & {"answers", "gold", "gold_evidence", "labels", "metrics", "evaluation"}


def test_reciprocal_neighbor_budget_dedup_order_and_expansion_cap():
    result, *_ = run_case("Who directed Alex Smith?", "Alex Smith acted in a film.",
                          [("c", "Alex Smith was directed by Dana Lee.", [0.8, 0.6], 0)], token_budget=10,
                          rule=ExpansionRule(1, 1))
    assert result.expanded_chunk_ids == []
    assert result.candidate_decisions[0]["reason"] == "token_budget_exceeded"
    assert result.context_token_count <= result.token_budget
    assert result.candidate_decisions[0]["candidate_chunk_id"] == "c"
    no_expansions, *_ = run_case("Who wrote Arizona?", "Arizona was recorded by Mark Lindsay.",
        [("c", "Arizona was written by Kenny Young.", [0.8, 0.6], 0)],
        rule=ExpansionRule(1, 0))
    assert no_expansions.expanded_chunk_ids == []
    assert no_expansions.candidate_decisions[0]["reason"] == "expansion_limit_reached"
    duplicate, *_ = run_case("Who wrote Arizona?", "Arizona was written by Kenny Young.",
        [("c", "Arizona was written by Kenny Young.", [0.8, 0.6], 0)])
    assert duplicate.expanded_chunk_ids == []


def test_invalid_nonreciprocal_neighbor_is_rejected():
    seed = chunk("s", "Arizona was written by Kenny Young.", 100, next_="c")
    candidate = chunk("c", "Kenny Young wrote Arizona.", 200, prev=None)
    search = SearchResponse([SearchHit(seed, 0.9, 1)], np.array([1., 0.]), 0.1)
    result = expand_context_v3(search, "Who wrote Arizona?", "multi_hop", [seed, candidate],
                               {"s": np.array([1., 0.]), "c": np.array([1., 0.])}, TinyTokenizer(),
                               lambda *_: .9, {"multi_hop": ExpansionRule(1, 2)}, 100)
    assert result.expanded_chunk_ids == []
    assert result.candidate_decisions[0]["reason"] == "invalid_neighbor_link"


def test_decision_logs_are_deterministic_and_explain_acceptance_or_rejection():
    args = ("Who wrote Arizona?", "Arizona was recorded by Mark Lindsay.",
            [("c", "Arizona was written by Kenny Young.", [0.8, 0.6], 0)])
    first = run_case(*args)[0]; second = run_case(*args)[0]
    assert first.candidate_decisions == second.candidate_decisions
    row = first.candidate_decisions[0]
    assert row["reason"] and "candidate_facts" in row and "supporting_fact_chain" in row
    assert row["accepted"] is True
    assert first.context_text.startswith("Arizona was written by Kenny Young.")
