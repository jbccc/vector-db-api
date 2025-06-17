from __future__ import annotations

import logging
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import ChunkRepository, DocumentRepository
from app.models import Chunk, Document
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


class ChunkService:
    """Service for managing document chunks and their operations."""

    def __init__(self, db: Session) -> None:
        """Initialize the ChunkService with database session and repositories."""
        self.db = db
        self.chunk_repo = ChunkRepository()
        self.doc_repo = DocumentRepository()
        self.doc_service = DocumentService(db)

    def create_chunks(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """Create chunks for a document."""
        created_chunks = []
        for chunk in chunks:
            obj_to_create = {
                "content": chunk.content,
                "document_id": document.id,
                "sequence_number": chunk.sequence_number,
            }
            try:
                created_chunk = self.chunk_repo.create(self.db, obj_in=obj_to_create)
                created_chunks.append(created_chunk)
            except IntegrityError:
                self.db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Chunk with sequence number {chunk.sequence_number} already exists for this document.",
                )
        return created_chunks

    def list_chunks(self, document_id: UUID | None = None) -> list[Chunk]:
        """List chunks optionally filtered by document ID."""
        if document_id:
            return self.chunk_repo.get_all_by_document(self.db, document_id)
        return self.chunk_repo.get_all(self.db)

    def get_chunk(self, chunk_id: int) -> Chunk:
        """Get a chunk by its ID."""
        chunk = self.chunk_repo.get(self.db, id=chunk_id)
        if not chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk with ID {chunk_id} not found",
            )
        return chunk

    def delete_document_chunks(self, document_id: int) -> None:
        """Delete all chunks belonging to a specific document."""
        document = self.doc_repo.get(self.db, id=document_id, lock=True)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found",
            )

        chunks = self.chunk_repo.get_all_by_document(self.db, document_id)
        for chunk in chunks:
            self.chunk_repo.delete(self.db, id=chunk.id)
