from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.models import Chunk, SemanticSearchRequest, SemanticSearchResponse
from app.models import Chunk as PydanticChunk
from app.services import VectorDBService


@pytest.fixture
def mock_db():
    return Mock(spec=Session)


@pytest.fixture
def vector_db_service(mock_db):
    return VectorDBService(mock_db)


@pytest.fixture
def sample_search_request():
    return SemanticSearchRequest(query="test search query", library_id=uuid4(), top_k=5)


@pytest.fixture
def sample_retrieval_results():
    doc_id = uuid4()

    return [
        Chunk(
            id=uuid4(),
            content="This is a test chunk about quick brown fox",
            sequence_number=0,
            document_id=doc_id,
            embedding=[0.1, 0.2, 0.3],
        ),
        Chunk(
            id=uuid4(),
            content="Another test chunk about lazy dog",
            sequence_number=1,
            document_id=doc_id,
            embedding=[0.4, 0.5, 0.6],
        ),
    ]


@pytest.fixture
def sample_document_id():
    return uuid4()


@pytest.fixture
def sample_library_id():
    return uuid4()


@pytest.fixture
def sample_chunks():
    doc_id = uuid4()
    return [
        Mock(
            id=uuid4(),
            content="Chunk 1 content",
            sequence_number=0,
            document_id=doc_id,
        ),
        Mock(
            id=uuid4(),
            content="Chunk 2 content",
            sequence_number=1,
            document_id=doc_id,
        ),
        Mock(
            id=uuid4(),
            content="Chunk 3 content",
            sequence_number=2,
            document_id=doc_id,
        ),
    ]


@pytest.fixture
def sample_documents():
    return [
        Mock(id=uuid4(), title="Document 1"),
        Mock(id=uuid4(), title="Document 2"),
        Mock(id=uuid4(), title="Document 3"),
    ]


def test_semantic_search_success(
    vector_db_service,
    sample_search_request,
    sample_retrieval_results,
) -> None:
    """Test successful semantic search."""
    vector_db_service.vector_store.retrieve_chunks = Mock(
        return_value=sample_retrieval_results,
    )

    result = vector_db_service.semantic_search(sample_search_request)

    assert isinstance(result, SemanticSearchResponse)
    assert result.results == sample_retrieval_results
    assert len(result.results) == 2

    vector_db_service.vector_store.retrieve_chunks.assert_called_once_with(
        vector_db_service.db,
        query=sample_search_request.query,
        library_id=sample_search_request.library_id,
        top_k=sample_search_request.top_k,
    )


def test_semantic_search_empty_results(
    vector_db_service, sample_search_request
) -> None:
    """Test semantic search with no results."""
    vector_db_service.vector_store.retrieve_chunks = Mock(return_value=[])

    result = vector_db_service.semantic_search(sample_search_request)

    assert isinstance(result, SemanticSearchResponse)
    assert result.results == []
    assert len(result.results) == 0


def test_semantic_search_with_default_top_k(vector_db_service) -> None:
    """Test semantic search uses default top_k when not specified."""
    request = SemanticSearchRequest(query="test query", library_id=uuid4())

    vector_db_service.vector_store.retrieve_chunks = Mock(return_value=[])

    vector_db_service.semantic_search(request)

    vector_db_service.vector_store.retrieve_chunks.assert_called_once_with(
        vector_db_service.db,
        query=request.query,
        library_id=request.library_id,
        top_k=request.top_k,
    )


def test_semantic_search_specific_library(
    vector_db_service, sample_retrieval_results
) -> None:
    """Test semantic search in a specific library."""
    library_id = uuid4()
    request = SemanticSearchRequest(
        query="specific library search",
        library_id=library_id,
        top_k=3,
    )

    vector_db_service.vector_store.retrieve_chunks = Mock(
        return_value=sample_retrieval_results,
    )

    result = vector_db_service.semantic_search(request)

    assert isinstance(result, SemanticSearchResponse)
    vector_db_service.vector_store.retrieve_chunks.assert_called_once_with(
        vector_db_service.db,
        query="specific library search",
        library_id=library_id,
        top_k=3,
    )


def test_semantic_search_preserves_ranking(vector_db_service) -> None:
    """Test that semantic search preserves the ranking from vector store."""
    results_with_chunks = [
        Chunk(
            id=uuid4(),
            content="High score chunk",
            sequence_number=0,
            document_id=uuid4(),
        ),
        Chunk(
            id=uuid4(),
            content="Medium score chunk",
            sequence_number=1,
            document_id=uuid4(),
        ),
        Chunk(
            id=uuid4(),
            content="Low score chunk",
            sequence_number=2,
            document_id=uuid4(),
        ),
    ]

    request = SemanticSearchRequest(query="test", library_id=uuid4(), top_k=3)
    vector_db_service.vector_store.retrieve_chunks = Mock(
        return_value=results_with_chunks,
    )

    result = vector_db_service.semantic_search(request)

    assert len(result.results) == 3
    assert result.results[0].id == results_with_chunks[0].id
    assert result.results[1].id == results_with_chunks[1].id
    assert result.results[2].id == results_with_chunks[2].id


def test_semantic_search() -> None:
    mock_db_session = MagicMock()

    with patch(
        "app.services.vector_db_service.get_vector_store",
    ) as mock_get_vector_store:
        mock_vector_store_instance = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store_instance

        vector_db_service = VectorDBService(db=mock_db_session)

        request = SemanticSearchRequest(
            query="test query",
            library_id=uuid4(),
            top_k=5,
        )

        mock_chunks = [
            PydanticChunk(
                id=uuid4(),
                document_id=uuid4(),
                content="Test chunk 1",
                sequence_number=0,
            ),
            PydanticChunk(
                id=uuid4(),
                document_id=uuid4(),
                content="Test chunk 2",
                sequence_number=1,
            ),
        ]
        mock_vector_store_instance.retrieve_chunks.return_value = mock_chunks

        response = vector_db_service.semantic_search(request)

        mock_vector_store_instance.retrieve_chunks.assert_called_once_with(
            mock_db_session,
            query=request.query,
            library_id=request.library_id,
            top_k=request.top_k,
        )
        assert len(response.results) == 2


def test_index_document_success(
    vector_db_service,
    sample_document_id,
    sample_library_id,
    sample_chunks,
) -> None:
    """Test successful document indexing."""
    vector_db_service.library_repo.get_by_document = Mock()
    vector_db_service.chunk_repo.get_all_by_document = Mock(return_value=sample_chunks)
    vector_db_service.document_repo.get = Mock(
        return_value=Mock(library_id=sample_library_id),
    )
    vector_db_service.vector_store.add_texts_batch = Mock()

    vector_db_service.index_document(sample_document_id)

    vector_db_service.library_repo.get_by_document.assert_called_once_with(
        vector_db_service.db,
        sample_document_id,
    )
    vector_db_service.chunk_repo.get_all_by_document.assert_called_once_with(
        vector_db_service.db,
        sample_document_id,
    )
    vector_db_service.document_repo.get.assert_called_once_with(
        vector_db_service.db,
        id=sample_document_id,
    )

    expected_chunks_dict = {str(chunk.id): chunk.content for chunk in sample_chunks}
    vector_db_service.vector_store.add_texts_batch.assert_called_once_with(
        sample_library_id,
        expected_chunks_dict,
    )


def test_index_document_empty_chunks(
    vector_db_service,
    sample_document_id,
    sample_library_id,
) -> None:
    """Test indexing document with no chunks."""
    vector_db_service.library_repo.get_by_document = Mock()
    vector_db_service.chunk_repo.get_all_by_document = Mock(return_value=[])
    vector_db_service.document_repo.get = Mock(
        return_value=Mock(library_id=sample_library_id),
    )
    vector_db_service.vector_store.add_texts_batch = Mock()

    vector_db_service.index_document(sample_document_id)

    vector_db_service.vector_store.add_texts_batch.assert_called_once_with(
        sample_library_id,
        {},
    )


def test_delete_document_index_success(vector_db_service, sample_document_id) -> None:
    """Test successful document index deletion."""
    vector_db_service.vector_store.delete = Mock()
    vector_db_service.document_service.mark_document_as_not_indexed = Mock()

    vector_db_service.delete_document_index(sample_document_id)

    vector_db_service.vector_store.delete.assert_called_once_with(
        str(sample_document_id),
    )
    vector_db_service.document_service.mark_document_as_not_indexed.assert_called_once_with(
        sample_document_id,
    )


def test_reindex_library_success(
    vector_db_service,
    sample_library_id,
    sample_documents,
) -> None:
    """Test successful library reindexing."""
    vector_db_service.library_repo.get = Mock()
    vector_db_service.document_repo.get_all_by_library = Mock(
        return_value=sample_documents,
    )
    vector_db_service.index_document = Mock()

    vector_db_service.reindex_library(sample_library_id)

    vector_db_service.library_repo.get.assert_called_once_with(
        vector_db_service.db,
        id=sample_library_id,
    )

    vector_db_service.document_repo.get_all_by_library.assert_called_once_with(
        vector_db_service.db,
        sample_library_id,
    )

    assert vector_db_service.index_document.call_count == len(sample_documents)
    for document in sample_documents:
        vector_db_service.index_document.assert_any_call(document.id)


def test_reindex_library_empty_library(vector_db_service, sample_library_id) -> None:
    """Test reindexing library with no documents."""
    vector_db_service.library_repo.get = Mock()
    vector_db_service.document_repo.get_all_by_library = Mock(return_value=[])
    vector_db_service.index_document = Mock()

    vector_db_service.reindex_library(sample_library_id)

    vector_db_service.index_document.assert_not_called()


def test_delete_library_index_success(
    vector_db_service,
    sample_library_id,
    sample_documents,
) -> None:
    """Test successful library index deletion."""
    vector_db_service.library_repo.get_documents = Mock(return_value=sample_documents)
    vector_db_service.delete_document_index = Mock()

    vector_db_service.delete_library_index(sample_library_id)

    vector_db_service.library_repo.get_documents.assert_called_once_with(
        vector_db_service.db,
        sample_library_id,
    )

    assert vector_db_service.delete_document_index.call_count == len(sample_documents)
    for document in sample_documents:
        vector_db_service.delete_document_index.assert_any_call(document.id)


def test_delete_library_index_empty_library(
    vector_db_service, sample_library_id
) -> None:
    """Test deleting library index with no documents."""
    vector_db_service.library_repo.get_documents = Mock(return_value=[])
    vector_db_service.delete_document_index = Mock()

    vector_db_service.delete_library_index(sample_library_id)

    vector_db_service.delete_document_index.assert_not_called()


def test_index_document_integration() -> None:
    """Integration test for document indexing with proper mocking."""
    mock_db_session = MagicMock()
    document_id = uuid4()
    library_id = uuid4()

    with (
        patch(
            "app.services.vector_db_service.get_vector_store",
        ) as mock_get_vector_store,
        patch(
            "app.services.vector_db_service.DocumentRepository",
        ) as mock_doc_repo_class,
        patch(
            "app.services.vector_db_service.LibraryRepository",
        ) as mock_lib_repo_class,
        patch(
            "app.services.vector_db_service.ChunkRepository",
        ) as mock_chunk_repo_class,
        patch(
            "app.services.vector_db_service.DocumentService",
        ) as mock_doc_service_class,
    ):
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store

        mock_doc_repo = MagicMock()
        mock_doc_repo_class.return_value = mock_doc_repo

        mock_lib_repo = MagicMock()
        mock_lib_repo_class.return_value = mock_lib_repo

        mock_chunk_repo = MagicMock()
        mock_chunk_repo_class.return_value = mock_chunk_repo

        mock_doc_service = MagicMock()
        mock_doc_service_class.return_value = mock_doc_service

        mock_chunks = [
            Mock(id=uuid4(), content="Content 1", sequence_number=0),
            Mock(id=uuid4(), content="Content 2", sequence_number=1),
        ]
        mock_document = Mock(library_id=library_id)

        mock_chunk_repo.get_all_by_document.return_value = mock_chunks
        mock_doc_repo.get.return_value = mock_document

        vector_db_service = VectorDBService(mock_db_session)
        vector_db_service.index_document(document_id)

        mock_lib_repo.get_by_document.assert_called_once_with(
            mock_db_session,
            document_id,
        )
        mock_chunk_repo.get_all_by_document.assert_called_once_with(
            mock_db_session,
            document_id,
        )
        mock_doc_repo.get.assert_called_once_with(mock_db_session, id=document_id)

        expected_chunks_dict = {str(chunk.id): chunk.content for chunk in mock_chunks}
        mock_vector_store.add_texts_batch.assert_called_once_with(
            library_id,
            expected_chunks_dict,
        )


def test_delete_document_index_integration() -> None:
    """Integration test for document index deletion with proper mocking."""
    mock_db_session = MagicMock()
    document_id = uuid4()

    with (
        patch(
            "app.services.vector_db_service.get_vector_store",
        ) as mock_get_vector_store,
        patch(
            "app.services.vector_db_service.DocumentService",
        ) as mock_doc_service_class,
    ):
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store

        mock_doc_service = MagicMock()
        mock_doc_service_class.return_value = mock_doc_service

        vector_db_service = VectorDBService(mock_db_session)
        vector_db_service.delete_document_index(document_id)

        mock_vector_store.delete.assert_called_once_with(str(document_id))
        mock_doc_service.mark_document_as_not_indexed.assert_called_once_with(
            document_id,
        )


def test_reindex_library_integration() -> None:
    """Integration test for library reindexing with proper mocking."""
    mock_db_session = MagicMock()
    library_id = uuid4()

    with (
        patch(
            "app.services.vector_db_service.get_vector_store",
        ) as mock_get_vector_store,
        patch(
            "app.services.vector_db_service.DocumentRepository",
        ) as mock_doc_repo_class,
        patch(
            "app.services.vector_db_service.LibraryRepository",
        ) as mock_lib_repo_class,
    ):
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store

        mock_doc_repo = MagicMock()
        mock_doc_repo_class.return_value = mock_doc_repo

        mock_lib_repo = MagicMock()
        mock_lib_repo_class.return_value = mock_lib_repo

        mock_documents = [Mock(id=uuid4()), Mock(id=uuid4())]
        mock_doc_repo.get_all_by_library.return_value = mock_documents

        vector_db_service = VectorDBService(mock_db_session)

        vector_db_service.index_document = Mock()

        vector_db_service.reindex_library(library_id)

        mock_lib_repo.get.assert_called_once_with(mock_db_session, id=library_id)
        mock_doc_repo.get_all_by_library.assert_called_once_with(
            mock_db_session,
            library_id,
        )

        assert vector_db_service.index_document.call_count == len(mock_documents)
        for doc in mock_documents:
            vector_db_service.index_document.assert_any_call(doc.id)


def test_delete_library_index_integration() -> None:
    """Integration test for library index deletion with proper mocking."""
    mock_db_session = MagicMock()
    library_id = uuid4()

    with (
        patch(
            "app.services.vector_db_service.get_vector_store",
        ) as mock_get_vector_store,
        patch(
            "app.services.vector_db_service.LibraryRepository",
        ) as mock_lib_repo_class,
    ):
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store

        mock_lib_repo = MagicMock()
        mock_lib_repo_class.return_value = mock_lib_repo

        mock_documents = [Mock(id=uuid4()), Mock(id=uuid4())]
        mock_lib_repo.get_documents.return_value = mock_documents

        vector_db_service = VectorDBService(mock_db_session)

        vector_db_service.delete_document_index = Mock()

        vector_db_service.delete_library_index(library_id)

        mock_lib_repo.get_documents.assert_called_once_with(mock_db_session, library_id)

        assert vector_db_service.delete_document_index.call_count == len(mock_documents)
        for doc in mock_documents:
            vector_db_service.delete_document_index.assert_any_call(doc.id)


def test_index_document_handles_vector_store_error(
    vector_db_service,
    sample_document_id,
) -> None:
    """Test that index_document properly handles vector store errors."""
    vector_db_service.library_repo.get_by_document = Mock()
    vector_db_service.chunk_repo.get_all_by_document = Mock(return_value=[])
    vector_db_service.document_repo.get = Mock(return_value=Mock(library_id=uuid4()))
    vector_db_service.vector_store.add_texts_batch = Mock(
        side_effect=Exception("Vector store error"),
    )

    with pytest.raises(Exception, match="Vector store error"):
        vector_db_service.index_document(sample_document_id)


def test_delete_document_index_handles_vector_store_error(
    vector_db_service,
    sample_document_id,
) -> None:
    """Test that delete_document_index handles vector store errors gracefully."""
    vector_db_service.vector_store.delete = Mock(side_effect=Exception("Delete error"))
    vector_db_service.document_service.mark_document_as_not_indexed = Mock()

    with pytest.raises(Exception, match="Delete error"):
        vector_db_service.delete_document_index(sample_document_id)
