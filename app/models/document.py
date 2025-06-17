from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import UUID

from pydantic import Field

from app.models import Chunk
from app.models.base import BaseEntity


class ProcessingStatus(str, Enum):
    """Document processing status."""

    NOT_INDEXED = "not_indexed"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(BaseEntity):
    """Document schema extending base content entity."""

    title: str = Field(..., min_length=1, max_length=500)
    library_id: UUID
    content: str
    chunks: list[Chunk] = Field(default_factory=list)

    processed_at: datetime | None = None
    processing_status: ProcessingStatus = ProcessingStatus.NOT_INDEXED

    @property
    def is_processed(self) -> bool:
        """Check if document is fully processed."""
        return self.processing_status == ProcessingStatus.INDEXED

    def mark_as_indexed(self) -> None:
        """Mark document as indexed."""
        self.processing_status = ProcessingStatus.INDEXED
        self.processed_at = datetime.now(UTC)
        self.update_timestamp()

    def mark_as_failed(self) -> None:
        """Mark document processing as failed."""
        self.processing_status = ProcessingStatus.FAILED
        self.processed_at = None
        self.update_timestamp()

    def mark_as_not_indexed(self) -> None:
        """Mark document processing as not indexed."""
        self.processing_status = ProcessingStatus.NOT_INDEXED
        self.processed_at = None
        self.update_timestamp()

    def mark_as_indexing(self) -> None:
        """Mark document processing as indexing."""
        self.processing_status = ProcessingStatus.INDEXING
        self.processed_at = None
        self.update_timestamp()
