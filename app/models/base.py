import uuid
from abc import ABC
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(UTC)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BaseEntity(TimestampMixin, ABC):
    """Abstract base class for all entities."""

    id: UUID = Field(default_factory=uuid.uuid4)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Create instance from dictionary."""
        return cls(**data)

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}
        validate_assignment = True
        from_attributes = True
