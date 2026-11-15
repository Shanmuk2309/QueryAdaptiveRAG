from pydantic import Field, field_validator
from .common import APIModel


class QueryRequest(APIModel):
    query: str = Field(min_length=1, max_length=10000)
    method: str
    top_k: int | None = Field(default=None, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class ChunkResult(APIModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float | None = None
    rank: int | None = None
    start: int
    end: int


class QueryMetrics(APIModel):
    retrieval_latency_ms: float
    expansion_latency_ms: float
    context_tokens: int
    retrieved_chunk_count: int
    expanded_chunk_count: int


class QueryResponse(APIModel):
    query: str
    method: str
    predicted_complexity: str | None
    retrieved_chunks: list[ChunkResult]
    expanded_chunks: list[ChunkResult]
    final_context: str
    sources: list[dict]
    metrics: QueryMetrics
    candidate_decisions: list[dict] = Field(default_factory=list)
