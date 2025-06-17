from uuid import UUID

from sqlalchemy.orm import Session

from app.core import get_vector_store
from app.db import ChunkRepository, DocumentRepository, LibraryRepository
from app.models import (
    Chunk,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.services import DocumentService


class VectorDBService:
    """Service for managing vector database operations including indexing, searching, and deletion."""

    def __init__(self, db: Session) -> None:
        """Initialize the VectorDBService with database session and repositories."""
        self.db = db
        self.vector_store = get_vector_store()
        self.document_repo = DocumentRepository()
        self.library_repo = LibraryRepository()
        self.chunk_repo = ChunkRepository()
        self.document_service = DocumentService(db)

    def semantic_search(self, request: SemanticSearchRequest) -> SemanticSearchResponse:
        """Perform semantic search on vector embeddings to find relevant chunks."""
        retrieval_results = self.vector_store.retrieve_chunks(
            self.db,
            query=request.query,
            library_id=request.library_id,
            top_k=request.top_k,
        )
        pydantic_chunks = [
            Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                sequence_number=chunk.sequence_number,
                created_at=chunk.created_at,
                updated_at=chunk.updated_at,
            )
            for chunk in retrieval_results
        ]
        return SemanticSearchResponse(results=pydantic_chunks)

    def reindex_library(self, library_id: UUID) -> None:
        """Reindex all documents in a library by removing and re-adding them to the vector store."""
        self.library_repo.get(self.db, id=library_id)
        documents = self.document_repo.get_all_by_library(self.db, library_id)
        for document in documents:
            self.index_document(document.id)

    def index_document(self, document_id: UUID) -> None:
        """Index a single document by adding its chunks to the vector store."""
        self.library_repo.get_by_document(self.db, document_id)
        chunks = self.chunk_repo.get_all_by_document(self.db, document_id)
        chunks_dict = {str(chunk.id): chunk.content for chunk in chunks}
        document = self.document_repo.get(self.db, id=document_id)
        self.vector_store.add_texts_batch(document.library_id, chunks_dict)

    def delete_document_index(self, document_id: UUID) -> None:
        """Remove document from vector index and mark it as NOT_INDEXED."""
        self.vector_store.delete(str(document_id))
        self.document_service.mark_document_as_not_indexed(document_id)

    def delete_library_index(self, library_id: UUID) -> None:
        """Delete the entire library index by removing all its documents from the vector store."""
        documents = self.library_repo.get_documents(self.db, library_id)
        for document in documents:
            self.delete_document_index(document.id)
