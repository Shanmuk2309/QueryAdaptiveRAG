from pydantic import Field, field_validator
from .common import APIModel
from .query import QueryResponse


class ComparisonRequest(APIModel):
    query: str = Field(min_length=1, max_length=10000)
    methods: list[str] = Field(min_length=1, max_length=4)
    top_k: int | None = Field(default=None, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class ComparisonResponse(APIModel):
    query: str
    results: dict[str, QueryResponse | dict]
