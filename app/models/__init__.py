"""Pydantic models for the document management system."""

from app.models.api import (
    ChunkCreate,
    DocumentCreate,
    DocumentUpdate,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.models.base import BaseEntity
from app.models.chunk import Chunk
from app.models.document import Document, ProcessingStatus
from app.models.library import Library
from app.models.vector import SearchResult

__all__ = [
    # Base classes
    "BaseEntity",
    "Chunk",
    "ChunkCreate",
    "Document",
    # Api models
    "DocumentCreate",
    "DocumentUpdate",
    "SearchResult",
    # Core models
    "Library",
    # Enums
    "ProcessingStatus",
    "SearchResult",
    "SemanticSearchRequest",
    "SemanticSearchResponse",
]
