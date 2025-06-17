from __future__ import annotations

from pydantic import Field

from app.models.base import BaseEntity


class Library(BaseEntity):
    """Library schema extending base content entity."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=1000)
