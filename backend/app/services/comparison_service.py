from ..utils.errors import BackendError
from .retrieval_service import RetrievalService


class ComparisonService:
    def __init__(self, retrieval: RetrievalService):
        self.retrieval = retrieval

    def compare(self, query: str, methods: list[str], top_k: int) -> dict:
        if len(set(methods)) != len(methods):
            raise BackendError(422, "duplicate_methods", "Each method may appear only once.")
        results = {}
        for method in methods:
            try:
                results[method] = self.retrieval.query(query, method, top_k)
            except BackendError as exc:
                if exc.status_code != 404:
                    raise
                results[method] = {"error": exc.detail}
        return {"query": query, "results": results}
