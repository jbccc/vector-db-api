"""Vector-related Pydantic models."""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a vector search result with similarity score."""

    vector_id: str = Field(..., description="The unique identifier of the vector")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between 0.0 and 1.0",
    )

    class Config:
        from_attributes = True
