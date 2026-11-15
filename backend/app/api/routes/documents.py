from fastapi import APIRouter, File, Request, UploadFile
from ...schemas.document import DocumentInfo, IndexRequest, IndexResponse, UploadResponse
from ...utils.validation import normalize_method
from ...utils.errors import BackendError

router = APIRouter()


def public(row: dict) -> dict:
    return {key: row[key] for key in ("document_id", "filename", "text_length", "processing_status", "indexed_methods")}


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload(
    request: Request,
    files: list[UploadFile] | None = File(None),
    file: UploadFile | None = File(None),
):
    # Keep the multi-file `files` field and also expose a direct binary `file`
    # field. Swagger UI consistently renders the latter as a native chooser;
    # both paths use the same validation, extraction, and storage service.
    uploads = [*(files or []), *([file] if file is not None else [])]
    rows = await request.app.state.document_service.upload(uploads)
    return {"documents": [public(row) for row in rows]}


@router.get("", response_model=list[DocumentInfo])
def list_documents(request: Request):
    return [public(row) for row in request.app.state.document_service.list()]


@router.get("/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str, request: Request):
    return public(request.app.state.document_service.get(document_id))


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request):
    request.app.state.document_service.delete(document_id)
    return None


@router.post("/index", response_model=IndexResponse)
def index_documents(body: IndexRequest, request: Request):
    method = normalize_method(body.method)
    service = request.app.state.indexing_service
    if service is None:
        raise BackendError(503, "embedding_unavailable", "The embedding runtime could not be initialized.")
    return service.index(method, body.document_ids)
