from uuid import UUID

from pydantic import Field

from app.models.base import BaseEntity


class Chunk(BaseEntity):
    """Chunk schema extending base entity."""

    document_id: UUID
    content: str = Field(..., min_length=1, max_length=10000)
    sequence_number: int = Field(..., ge=0)
