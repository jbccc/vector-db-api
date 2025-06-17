from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import LibraryRepository
from app.models import Document, Library


class LibraryService:
    """Service for managing library operations and CRUD functionality."""

    def __init__(self, db: Session) -> None:
        """Initialize the LibraryService with database session and repository."""
        self.db = db
        self.repo = LibraryRepository()

    def create_library(self, library_data: dict) -> Library:
        """Create a new library with the provided data."""
        try:
            return self.repo.create(self.db, obj_in=library_data)
        except IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Library with name '{library_data['name']}' already exists",
            )

    def list_libraries(self, search: str | None = None) -> list[Library]:
        """List all libraries with optional search filtering by name."""
        if search:
            return self.repo.search_by_name(self.db, query=search)
        return self.repo.get_all(self.db)

    def get_library(self, library_id: UUID, *, lock: bool = False) -> Library:
        """Get a library by ID with optional row locking."""
        library = self.repo.get(self.db, id=library_id, lock=lock)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID {library_id} not found",
            )
        return library

    def update_library(self, library_id: UUID, library_update: dict) -> Library:
        """Update an existing library with new data."""
        db_library = self.get_library(library_id, lock=True)

        if "name" in library_update and library_update["name"] != db_library.name:
            if self.repo.get_by_name(self.db, library_update["name"]):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Library with name '{library_update['name']}' already exists",
                )

        return self.repo.update(self.db, id=library_id, obj_in=library_update)

    def delete_library(self, library_id: UUID) -> None:
        """Delete a library by its ID."""
        self.get_library(library_id)
        self.repo.delete(self.db, id=library_id)

    def list_library_documents(self, library_id: UUID) -> list[Document]:
        """Get all documents belonging to a specific library."""
        self.get_library(library_id)
        db_documents = self.repo.get_documents(self.db, library_id)
        return [Document.model_validate(doc.to_dict()) for doc in db_documents]
