from __future__ import annotations

from sqlalchemy import UUID, Column, String, select
from sqlalchemy.orm import Session, relationship

from app.db.base import BaseModel
from app.db.repositories.base import BaseRepository
from app.db.repositories.document import Document


class Library(BaseModel):
    """Library model for storing document collections."""

    __tablename__ = "libraries"

    name = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    documents = relationship(
        "Document",
        back_populates="library",
        cascade="all, delete-orphan",
    )


class LibraryRepository(BaseRepository[Library]):
    """Repository for library operations."""

    def __init__(self) -> None:
        """Initialize the LibraryRepository."""
        super().__init__(Library)

    def get_by_name(self, db: Session, name: str) -> Library | None:
        """Get library by name."""
        stmt = select(Library).where(Library.name == name)
        return db.execute(stmt).scalar_one_or_none()

    def search_by_name(
        self,
        db: Session,
        query: str,
    ) -> list[Library]:
        """Search libraries by name."""
        stmt = select(Library).where(Library.name.ilike(f"%{query}%"))
        return list(db.execute(stmt).scalars().one_or_none())

    def get_by_document(
        self,
        db: Session,
        document_id: UUID,
    ) -> Library | None:
        """Search library by document."""
        stmt = select(Library).where(Library.documents.any(Document.id == document_id))
        return db.execute(stmt).scalar_one_or_none()

    def get_documents(self, db: Session, library_id: UUID) -> list[Document]:
        """Get all documents in a library."""
        stmt = select(Document).where(Document.library_id == library_id)
        return list(db.execute(stmt).scalars().all())
