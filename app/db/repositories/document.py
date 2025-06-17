from __future__ import annotations

from sqlalchemy import UUID, Column, DateTime, ForeignKey, String, select
from sqlalchemy.orm import Session, relationship

from app.db.base import BaseModel
from app.db.repositories.base import BaseRepository


class Document(BaseModel):
    """Document model for storing documents and their metadata."""

    __tablename__ = "documents"

    title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    library_id = Column(UUID(as_uuid=True), ForeignKey("libraries.id"))

    processing_status = Column(String, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    library = relationship("Library", back_populates="documents")
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentRepository(BaseRepository[Document]):
    """Repository for document operations."""

    def __init__(self) -> None:
        """Initialize the DocumentRepository."""
        super().__init__(Document)

    def get_by_title(
        self,
        db: Session,
        title: str,
        *,
        lock: bool = False,
    ) -> Document | None:
        """Get document by title."""
        stmt = select(Document).where(Document.title == title)
        if lock:
            stmt = stmt.with_for_update()
        return db.execute(stmt).scalar_one_or_none()

    def get_all_by_library(
        self,
        db: Session,
        library_id: UUID,
    ) -> list[Document]:
        """Get all documents in a library."""
        stmt = select(Document).where(Document.library_id == library_id)
        return list(db.execute(stmt).scalars().all())
