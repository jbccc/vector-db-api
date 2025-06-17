from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from app.api.deps import LibraryServiceDep
from app.models import Library

router = APIRouter()


@router.post("", response_model=Library, status_code=status.HTTP_201_CREATED)
async def create_library(library: Library, service: LibraryServiceDep) -> Library:
    """Create a new library."""
    library_data = {"name": library.name, "description": library.description}
    return service.create_library(library_data)


@router.get("", response_model=list[Library])
async def list_libraries(
    service: LibraryServiceDep,
    search: Annotated[
        str | None,
        Query(description="Search term for library name"),
    ] = None,
) -> list[Library]:
    """List all libraries with optional filtering."""
    return service.list_libraries(search)


@router.get("/{library_id}")
async def get_library(library_id: UUID, service: LibraryServiceDep) -> Library:
    """Get a specific library by ID."""
    return service.get_library(library_id)


@router.put("/{library_id}")
async def update_library(
    library_id: UUID,
    library: Library,
    service: LibraryServiceDep,
) -> Library:
    """Update a library."""
    update_data = library.model_dump(exclude_unset=True)
    return service.update_library(library_id, update_data)


@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(library_id: UUID, service: LibraryServiceDep):
    """Delete a library and all its documents/chunks."""
    service.delete_library(library_id)
