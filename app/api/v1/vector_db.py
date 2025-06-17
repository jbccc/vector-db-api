from uuid import UUID

from fastapi import APIRouter, status

from app.api.deps import DocumentServiceDep, LibraryServiceDep, VectorDBServiceDep
from app.models import (
    Document,
    SemanticSearchRequest,
    SemanticSearchResponse,
)

router = APIRouter()


@router.post(
    "/index/{document_id}",
    status_code=status.HTTP_201_CREATED,
)
async def index_document(
    document_id: UUID,
    vector_db_service: VectorDBServiceDep,
    library_service: LibraryServiceDep,
    document_service: DocumentServiceDep,
) -> Document:
    """Index a document with its chunks into the vector database.

    This endpoint:
    - Creates/updates the document in the library
    - Processes and stores chunks
    - Generates embeddings for vector search
    - Adds vectors to the search index
    """
    library_service.get_library(document_id)
    vector_db_service.index_document(document_id)
    return document_service.get_document(document_id)


@router.delete("/index/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_from_index(
    document_id: UUID,
    vector_db_service: VectorDBServiceDep,
    document_service: DocumentServiceDep,
) -> None:
    """Remove a document and its chunks from the vector index.

    This removes the vectors from the search index but keeps
    the document metadata in the database.
    """
    vector_db_service.delete_document_index(document_id)
    return document_service.get_document(document_id)


@router.post("/search")
async def vector_search(
    request: SemanticSearchRequest,
    vector_db_service: VectorDBServiceDep,
    library_service: LibraryServiceDep,
) -> SemanticSearchResponse:
    """Perform vector similarity search across documents in a library.

    Uses the configured vector index (BruteForce or LSH) to find
    the most semantically similar chunks to the query.
    """
    library_service.get_library(request.library_id)

    return vector_db_service.semantic_search(request)


@router.post("/reindex/{library_id}", status_code=status.HTTP_202_ACCEPTED)
async def reindex_library(
    library_id: UUID,
    vector_db_service: VectorDBServiceDep,
    library_service: LibraryServiceDep,
) -> dict[str, str]:
    """Reindex all documents in a library.

    This rebuilds the vector index for all documents in the library,
    useful after configuration changes or index corruption.
    """
    library_service.get_library(library_id)
    vector_db_service.reindex_library(library_id)
    return {"message": f"Reindexing library {library_id}"}


@router.delete("/index/library/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library_index(
    library_id: UUID,
    vector_db_service: VectorDBServiceDep,
) -> None:
    """Delete the vector index for a library."""
    vector_db_service.delete_library_index(library_id)
    return {"message": f"Removed library index for {library_id}"}
