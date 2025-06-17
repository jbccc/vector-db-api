from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.models.chunk import Chunk


class SemanticSearchRequest(BaseModel):
    """Semantic search request to find relevant chunks."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The natural language query to search for.",
    )
    library_id: UUID = Field(..., description="The ID of the library to search within.")
    top_k: int = Field(
        default=10,
        ge=1,
        description="The maximum number of results to return.",
    )
    algorithm: str = Field(
        default="knn",
        description="Algorithm to be used for the vector retrieval. Options are KKN and ANN.",
    )


class SemanticSearchResponse(BaseModel):
    """Semantic search response."""

    results: list[Chunk] = Field(default_factory=list)


class ChunkCreate(BaseModel):
    """Schema for creating a new chunk."""

    content: str = Field(..., min_length=1, max_length=10000)
    sequence_number: int = Field(..., ge=0)


class DocumentCreate(BaseModel):
    """Schema for creating a new document with chunks."""

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    chunks: list[ChunkCreate] = Field(..., min_length=1)
    index: bool = Field(False)


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""

    title: str | None = Field(None, min_length=1, max_length=500)
    content: str | None = Field(None, min_length=1)
