from __future__ import annotations

from time import perf_counter

from query_adaptive_rag.classification.complexity import classify_query
from query_adaptive_rag.expansion.context_expander import ExpansionRule, TokenBudgetError, assemble_context
from query_adaptive_rag.expansion.context_expander_v3 import expand_context_v3

from ..config import Settings
from ..utils.errors import BackendError
from .runtime import Runtime


EXPANSION_RULES = {
    "single_fact": ExpansionRule(0, 0),
    "multi_concept": ExpansionRule(1, 2),
    "multi_hop": ExpansionRule(2, 4),
}


class RetrievalService:
    def __init__(self, settings: Settings, runtime: Runtime):
        self.settings, self.runtime = settings, runtime

    @staticmethod
    def _view(chunk: dict, score=None, rank=None) -> dict:
        return {"chunk_id": chunk["chunk_id"], "document_id": chunk["doc_id"], "filename": chunk.get("filename", ""),
                "text": chunk["text"], "score": score, "rank": rank,
                "start": int(chunk.get("start", 0)), "end": int(chunk.get("end", 0))}

    def query(self, query: str, method: str, top_k: int) -> dict:
        retriever = self.runtime.indexes.get(method)
        if retriever is None:
            raise BackendError(404, "index_not_found", f"No index is available for {method}.")
        try:
            search = retriever.retrieve(query.strip(), top_k)
        except ValueError as exc:
            raise BackendError(422, "invalid_query", str(exc)) from exc
        except Exception as exc:
            raise BackendError(500, "retrieval_failed", "Query embedding or FAISS retrieval failed.") from exc
        hits = search.hits
        retrieved = [self._view(hit.chunk, hit.score, hit.rank) for hit in hits]
        expanded: list[dict] = []
        candidate_decisions = []
        classification = classify_query(query)
        complexity = classification["complexity"]
        expansion_ms = 0.0
        try:
            if method == "M4-v3":
                result = expand_context_v3(search, query, complexity, self.runtime.chunks_by_method[method],
                    self.runtime.vectors_by_method[method], self.runtime.tokenizer, retriever.score,
                    EXPANSION_RULES, self.settings.context_token_budget)
                by_id = {chunk["chunk_id"]: chunk for chunk in self.runtime.chunks_by_method[method]}
                expanded = [self._view(by_id[chunk_id], result.similarity_scores.get(chunk_id)) for chunk_id in result.expanded_chunk_ids]
                final_context, context_tokens = result.context_text, result.context_token_count
                expansion_ms = result.latency_ms
                candidate_decisions = result.candidate_decisions
            else:
                final_context, context_tokens = assemble_context([hit.chunk for hit in hits], self.runtime.tokenizer)
                if context_tokens > self.settings.context_token_budget:
                    raise BackendError(413, "context_budget_exceeded", "Retrieved context exceeds the configured token budget; lower top_k.")
        except TokenBudgetError as exc:
            raise BackendError(413, "context_budget_exceeded", str(exc)) from exc
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(500, "retrieval_failed", "Query retrieval or context expansion failed.") from exc
        source_ids = list(dict.fromkeys(hit.chunk["doc_id"] for hit in hits))
        source_by_doc = {chunk["doc_id"]: {"document_id": chunk["doc_id"], "filename": chunk.get("filename", "")} for hit in hits for chunk in [hit.chunk]}
        return {"query": query, "method": method, "predicted_complexity": complexity if method == "M4-v3" else None,
                "retrieved_chunks": retrieved, "expanded_chunks": expanded, "final_context": final_context,
                "sources": [source_by_doc[doc_id] for doc_id in source_ids],
                "metrics": {"retrieval_latency_ms": search.latency_ms, "expansion_latency_ms": expansion_ms,
                            "context_tokens": context_tokens, "retrieved_chunk_count": len(retrieved),
                            "expanded_chunk_count": len(expanded)}, "candidate_decisions": candidate_decisions}
