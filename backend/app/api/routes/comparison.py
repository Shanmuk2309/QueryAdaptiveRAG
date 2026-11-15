from fastapi import APIRouter, Request
from ...schemas.comparison import ComparisonRequest, ComparisonResponse
from ...utils.validation import normalize_method
from ...utils.errors import BackendError

router = APIRouter()


@router.post("", response_model=ComparisonResponse)
def compare(body: ComparisonRequest, request: Request):
    methods = [normalize_method(method) for method in body.methods]
    top_k = body.top_k or request.app.state.settings.top_k_default
    service = request.app.state.comparison_service
    if service is None:
        raise BackendError(503, "embedding_unavailable", "The embedding runtime could not be initialized.")
    return service.compare(body.query.strip(), methods, top_k)
