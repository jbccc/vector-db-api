from sqlalchemy import UUID, Column, ForeignKey, Integer, String, select
from sqlalchemy.orm import Session, relationship

from app.db.base import BaseModel
from app.db.repositories.base import BaseRepository
from app.db.repositories.document import Document


class Chunk(BaseModel):
    """Chunk model for storing document chunks."""

    __tablename__ = "chunks"

    content = Column(String, nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    document = relationship("Document", back_populates="chunks")
    sequence_number = Column(Integer, nullable=False, comment="Position in document")


class ChunkRepository(BaseRepository[Chunk]):
    """Repository for chunk operations."""

    def __init__(self) -> None:
        """Initialize the ChunkRepository."""
        super().__init__(Chunk)

    def get_all_by_document(self, db: Session, document_id: int) -> list[Chunk]:
        """Get all chunks for a document."""
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.sequence_number)
        )
        return list(db.execute(stmt).scalars().all())

    def get_all_by_library(self, db: Session, library_id: UUID) -> list[Chunk]:
        """Get all chunks for a library."""
        stmt = select(Chunk).join(Document).where(Document.library_id == library_id)
        return list(db.execute(stmt).scalars().all())

    def get_by_ids(self, db: Session, chunk_ids: list[str]) -> list[Chunk]:
        """Get chunks by their IDs."""
        if not chunk_ids:
            return []
        stmt = select(Chunk).where(Chunk.id.in_(chunk_ids))
        return list(db.execute(stmt).scalars().all())
