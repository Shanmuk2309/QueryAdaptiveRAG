from pydantic import Field
from .common import APIModel


class DocumentInfo(APIModel):
    document_id: str
    filename: str
    text_length: int
    upload_status: str = "stored"
    processing_status: str
    indexed_methods: list[str] = Field(default_factory=list)


class UploadResponse(APIModel):
    documents: list[DocumentInfo]


class IndexRequest(APIModel):
    method: str
    document_ids: list[str] | None = None


class IndexResponse(APIModel):
    method: str
    document_count: int
    chunk_count: int
    embedding_model: str
    indexing_duration_ms: float
    index_status: str
