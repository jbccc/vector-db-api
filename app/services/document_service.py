from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import DocumentRepository
from app.models import Document, ProcessingStatus


class DocumentService:
    """Service for managing document operations and processing status updates."""

    def __init__(self, db: Session) -> None:
        """Initialize the DocumentService with database session and repository."""
        self.db = db
        self.doc_repo = DocumentRepository()

    def create_document(self, document_data: dict) -> Document:
        """Create a document."""
        # Ensure processing_status is set to default if not provided
        if "processing_status" not in document_data:
            document_data["processing_status"] = ProcessingStatus.NOT_INDEXED.value

        try:
            return self.doc_repo.create(self.db, obj_in=document_data)
        except IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document with title '{document_data['title']}' already exists",
            )

    def get_document(self, document_id: UUID, *, lock: bool = False) -> Document:
        """Get a document and optionally lock it for updates."""
        document = self.doc_repo.get(self.db, id=document_id, lock=lock)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found",
            )
        return document

    def mark_document_as_indexing(self, document_id: UUID) -> None:
        """Mark document as indexing using the Document model method."""
        db_document = self.get_document(document_id, lock=True)
        pydantic_doc = Document.model_validate(db_document.to_dict())
        pydantic_doc.mark_as_indexing()

        self.doc_repo.update(
            self.db,
            id=document_id,
            obj_in={
                "processing_status": pydantic_doc.processing_status,
                "processed_at": pydantic_doc.processed_at,
            },
        )

    def mark_document_as_indexed(self, document_id: UUID) -> None:
        """Mark document as indexed using the Document model method."""
        db_document = self.get_document(document_id, lock=True)
        pydantic_doc = Document.model_validate(db_document.to_dict())
        pydantic_doc.mark_as_indexed()

        self.doc_repo.update(
            self.db,
            id=document_id,
            obj_in={
                "processing_status": pydantic_doc.processing_status,
                "processed_at": pydantic_doc.processed_at,
            },
        )

    def mark_document_as_failed(self, document_id: UUID) -> None:
        """Mark document as failed using the Document model method."""
        db_document = self.get_document(document_id, lock=True)
        pydantic_doc = Document.model_validate(db_document.to_dict())
        pydantic_doc.mark_as_failed()

        self.doc_repo.update(
            self.db,
            id=document_id,
            obj_in={
                "processing_status": pydantic_doc.processing_status,
                "processed_at": pydantic_doc.processed_at,
            },
        )

    def mark_document_as_not_indexed(self, document_id: UUID) -> None:
        """Mark document as not indexed using the Document model method."""
        db_document = self.get_document(document_id, lock=True)
        pydantic_doc = Document.model_validate(db_document.to_dict())
        pydantic_doc.mark_as_not_indexed()

        self.doc_repo.update(
            self.db,
            id=document_id,
            obj_in={
                "processing_status": pydantic_doc.processing_status,
                "processed_at": pydantic_doc.processed_at,
            },
        )

    def update_document(self, document_id: UUID, update_data: dict) -> Document:
        """Update a document."""
        self.get_document(document_id, lock=True)

        try:
            return self.doc_repo.update(self.db, id=document_id, obj_in=update_data)
        except IntegrityError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A document with the provided attributes already exists.",
            )

    def list_documents(self, library_id: UUID) -> list[Document]:
        """List all documents in a library."""
        return self.doc_repo.get_all_by_library(self.db, library_id)

    def delete_document(self, document_id: UUID) -> None:
        """Delete a document by its ID."""
        self.get_document(document_id)
        self.doc_repo.delete(self.db, id=document_id)
