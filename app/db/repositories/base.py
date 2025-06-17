from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.db.base import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations."""

    def __init__(self, model: type[ModelType]) -> None:
        """Initialize the BaseRepository with a model type."""
        self.model = model

    def get(
        self,
        db: Session,
        id: int,
        *,
        lock: bool = False,
        skip_locked: bool = False,
    ) -> ModelType | None:
        """Get a single record by ID."""
        if lock:
            return db.get(
                self.model,
                id,
                with_for_update=True,
                **({"skip_locked": True} if skip_locked else {}),
            )
        return db.get(self.model, id)

    def get_all(self, db: Session) -> list[ModelType]:
        """Get all records."""
        stmt = select(self.model)
        return list(db.execute(stmt).scalars().all())

    def create(self, db: Session, *, obj_in: dict[str, Any]) -> ModelType:
        """Create a new record."""
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        db.flush()
        return db_obj

    def update(
        self,
        db: Session,
        *,
        id: int,
        obj_in: dict[str, Any],
        lock: bool = True,
    ) -> ModelType | None:
        """Update a record."""
        if lock:
            existing = self.get(db, id, lock=True)
            if not existing:
                return None

        stmt = (
            update(self.model)
            .where(self.model.id == id)
            .values(**obj_in)
            .execution_options(synchronize_session="fetch")
        )
        db.execute(stmt)
        return self.get(db, id)

    def delete(self, db: Session, *, id: int, lock: bool = True) -> bool:
        """Delete a record."""
        obj = self.get(db, id=id, lock=lock)
        if not obj:
            return False
        db.delete(obj)
        db.flush()
        return True

    def exists(self, db: Session, id: int) -> bool:
        """Check if a record exists."""
        return self.get(db, id) is not None

    def get_many(
        self,
        db: Session,
        ids: list[int],
        *,
        lock: bool = False,
    ) -> list[ModelType]:
        """Get multiple records by IDs."""
        stmt = select(self.model).where(self.model.id.in_(ids))
        if lock:
            stmt = stmt.with_for_update()
        return list(db.execute(stmt).scalars().all())

    def delete_many(self, db: Session, ids: list[int], *, lock: bool = True) -> int:
        """Delete multiple records by IDs."""
        if lock:
            locked_objects = self.get_many(db, ids, lock=True)
            existing_ids = [obj.id for obj in locked_objects]
        else:
            existing_ids = ids

        stmt = delete(self.model).where(self.model.id.in_(existing_ids))
        result = db.execute(stmt)
        db.flush()
        return result.rowcount
