import logging
from uuid import UUID

from fastapi import APIRouter, status

from app.api.deps import (
    ChunkServiceDep,
    DocumentServiceDep,
    LibraryServiceDep,
    VectorDBServiceDep,
)
from app.models import Chunk, Document, DocumentCreate, DocumentUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def list_documents_in_library(
    library_id: UUID,
    library_service: LibraryServiceDep,
    document_service: DocumentServiceDep,
) -> list[Document]:
    """List all documents in a library."""
    library_service.get_library(library_id)
    return document_service.list_documents(library_id)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_document(
    library_id: UUID,
    payload: DocumentCreate,
    library_service: LibraryServiceDep,
    document_service: DocumentServiceDep,
    chunk_service: ChunkServiceDep,
    vector_db_service: VectorDBServiceDep,
) -> Document:
    """Create a new document in a library (without indexing).

    To index the document for vector search, use /vector-db/index endpoint.
    """
    library_service.get_library(library_id)

    document_data = {
        "title": payload.title,
        "content": payload.content,
        "library_id": library_id,
    }

    document = document_service.create_document(document_data)

    if payload.chunks:
        chunks_to_create = [
            Chunk(
                document_id=document.id,
                content=chunk_create.content,
                sequence_number=chunk_create.sequence_number,
            )
            for chunk_create in payload.chunks
        ]

        chunk_service.create_chunks(chunks_to_create, document)

    if payload.index:
        document_service.mark_document_as_indexing(document.id)
        try:
            vector_db_service.index_document(document.id)
            document_service.mark_document_as_indexed(document.id)
        except Exception:
            document_service.mark_document_as_failed(document.id)
            raise

    return document


@router.get("/{document_id}")
async def get_document(
    library_id: UUID,
    document_id: UUID,
    library_service: LibraryServiceDep,
    document_service: DocumentServiceDep,
) -> Document:
    """Get a specific document by ID within a library."""
    library_service.get_library(library_id)
    return document_service.get_document(document_id)


@router.put("/{document_id}")
async def update_document(
    library_id: UUID,
    document_id: UUID,
    payload: DocumentUpdate,
    library_service: LibraryServiceDep,
    document_service: DocumentServiceDep,
) -> Document:
    """Update a document's metadata/content (without re-indexing).

    To re-index after updating, use /vector-db/index endpoint.
    """
    # Validate library exists
    library_service.get_library(library_id)

    # Update document
    update_data = payload.model_dump(exclude_unset=True)
    return document_service.update_document(document_id, update_data)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    library_id: UUID,
    document_id: UUID,
    library_service: LibraryServiceDep,
    document_service: DocumentServiceDep,
    chunk_service: ChunkServiceDep,
) -> None:
    """Delete a document and all its chunks from both DB and vector index."""
    library_service.get_library(library_id)
    document_service.get_document(document_id)
    document_service.delete_document(document_id)
    chunk_service.delete_document_chunks(document_id)


# CHUNK ACCESS IS READ-ONLY
@router.get("/{document_id}/chunks")
async def get_document_chunks(
    library_id: UUID,
    document_id: UUID,
    library_service: LibraryServiceDep,
    chunk_service: ChunkServiceDep,
) -> list[Chunk]:
    """Get all chunks for a specific document."""
    library_service.get_library(library_id)
    return chunk_service.list_chunks(document_id)


@router.get("/{document_id}/chunks/{chunk_id}")
async def get_document_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    library_service: LibraryServiceDep,
    chunk_service: ChunkServiceDep,
) -> Chunk:
    """Get a specific chunk within a document."""
    library_service.get_library(library_id)
    chunk = chunk_service.get_chunk(chunk_id)
    if chunk.document_id != document_id:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk {chunk_id} not found in document {document_id}",
        )

    return chunk
