from fastapi import APIRouter, Request
from ...schemas.query import QueryRequest, QueryResponse
from ...utils.validation import normalize_method
from ...utils.errors import BackendError

router = APIRouter()


@router.post("", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    method = normalize_method(body.method)
    top_k = body.top_k or request.app.state.settings.top_k_default
    service = request.app.state.retrieval_service
    if service is None:
        raise BackendError(503, "embedding_unavailable", "The embedding runtime could not be initialized.")
    return service.query(body.query.strip(), method, top_k)
