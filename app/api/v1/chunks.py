from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.api.deps import ChunkServiceDep
from app.config import settings
from app.models import Chunk

logger = logging.getLogger(__name__)

router = APIRouter()


# STANDALONE CHUNK OPERATIONS (debugging only)
# Note: In the normal API flow, chunks should be accessed through documents
# These endpoints are exclusively for debugging purposes


@router.get("")
async def list_all_chunks(
    service: ChunkServiceDep,
    document_id: UUID | None = None,
) -> list[Chunk]:
    """List all chunks, optionally filtered by document.

    WARNING: This is for debugging purposes only.
    In normal usage, access chunks through /libraries/{id}/documents/{id}/chunks
    """
    logger.warning(
        "This endpoint is for debugging purposes only. Use /libraries/{id}/documents/{id}/chunks instead.",
    )
    if settings.is_development:
        return service.list_chunks(document_id)
    raise HTTPException(
        status_code=403,
        detail="This endpoint is only available in development mode",
    )


@router.get("/{chunk_id}")
async def get_chunk_by_id(chunk_id: UUID, service: ChunkServiceDep) -> Chunk:
    """Get a chunk by ID directly.

    WARNING: This is for debugging purposes only.
    In normal usage, access chunks through /libraries/{id}/documents/{id}/chunks/{id}
    """
    logger.warning(
        "This endpoint is for debugging purposes only. Use /libraries/{id}/documents/{id}/chunks instead.",
    )

    if settings.is_development:
        return service.get_chunk(chunk_id)
    raise HTTPException(
        status_code=403,
        detail="This endpoint is only available in development mode",
    )
