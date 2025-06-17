from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db import ChunkRepository, DocumentRepository, LibraryRepository, get_db
from app.services import ChunkService, DocumentService, LibraryService, VectorDBService


def get_db_session() -> Generator[Session, None, None]:
    """Get database session."""
    with get_db() as db:
        yield db


def get_library_repo() -> LibraryRepository:
    """Get library repository."""
    return LibraryRepository()


def get_document_repo() -> DocumentRepository:
    """Get document repository."""
    return DocumentRepository()


def get_chunk_repo() -> ChunkRepository:
    """Get chunk repository."""
    return ChunkRepository()


def get_library_service(db: "DB") -> LibraryService:
    """Get library service."""
    return LibraryService(db)


def get_chunk_service(db: "DB") -> ChunkService:
    """Get chunk service."""
    return ChunkService(db)


def get_document_service(db: "DB") -> DocumentService:
    """Get document service."""
    return DocumentService(db)


def get_vector_db_service(db: "DB") -> VectorDBService:
    """Get vector db service."""
    return VectorDBService(db)


DB = Annotated[Session, Depends(get_db_session)]
LibraryRepo = Annotated[LibraryRepository, Depends(get_library_repo)]
DocumentRepo = Annotated[DocumentRepository, Depends(get_document_repo)]
ChunkRepo = Annotated[ChunkRepository, Depends(get_chunk_repo)]
LibraryServiceDep = Annotated[LibraryService, Depends(get_library_service)]
ChunkServiceDep = Annotated[ChunkService, Depends(get_chunk_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
VectorDBServiceDep = Annotated[VectorDBService, Depends(get_vector_db_service)]
