from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.services import ChunkService


@pytest.fixture
def mock_db():
    return Mock(spec=Session)


@pytest.fixture
def chunk_service(mock_db):
    return ChunkService(mock_db)


@pytest.fixture
def sample_document():
    return Document(
        id=uuid4(),
        title="Test Document",
        content="Test content",
        library_id=uuid4(),
        processing_status="indexed",
    )


@pytest.fixture
def sample_chunks():
    doc_id = uuid4()
    return [
        Chunk(
            id=uuid4(),
            content="First chunk content",
            sequence_number=0,
            document_id=doc_id,
        ),
        Chunk(
            id=uuid4(),
            content="Second chunk content",
            sequence_number=1,
            document_id=doc_id,
        ),
    ]


def test_create_chunks_success(chunk_service, sample_document, sample_chunks) -> None:
    """Test creating chunks for a document successfully."""
    mock_created_chunks = [Mock(id=uuid4()) for _ in sample_chunks]
    chunk_service.chunk_repo.create = Mock(side_effect=mock_created_chunks)

    result = chunk_service.create_chunks(sample_chunks, sample_document)

    assert len(result) == 2
    assert chunk_service.chunk_repo.create.call_count == 2

    # Verify the create calls had correct data
    create_calls = chunk_service.chunk_repo.create.call_args_list
    assert create_calls[0][1]["obj_in"]["content"] == "First chunk content"
    assert create_calls[0][1]["obj_in"]["sequence_number"] == 0
    assert create_calls[0][1]["obj_in"]["document_id"] == sample_document.id


def test_create_chunks_integrity_error(
    chunk_service, sample_document, sample_chunks
) -> None:
    """Test creating chunks with duplicate sequence number raises conflict."""
    from sqlalchemy.exc import IntegrityError

    chunk_service.chunk_repo.create = Mock(side_effect=IntegrityError(None, None, None))
    chunk_service.db.rollback = Mock()

    with pytest.raises(HTTPException) as exc_info:
        chunk_service.create_chunks(sample_chunks, sample_document)

    assert exc_info.value.status_code == 409
    assert "already exists" in exc_info.value.detail
    chunk_service.db.rollback.assert_called_once()


def test_list_chunks_all(chunk_service, sample_chunks) -> None:
    """Test listing all chunks."""
    chunk_service.chunk_repo.get_all = Mock(return_value=sample_chunks)

    result = chunk_service.list_chunks()

    assert result == sample_chunks
    chunk_service.chunk_repo.get_all.assert_called_once_with(chunk_service.db)


def test_list_chunks_by_document(chunk_service, sample_chunks) -> None:
    """Test listing chunks for a specific document."""
    document_id = uuid4()
    chunk_service.chunk_repo.get_all_by_document = Mock(return_value=sample_chunks)

    result = chunk_service.list_chunks(document_id=document_id)

    assert result == sample_chunks
    chunk_service.chunk_repo.get_all_by_document.assert_called_once_with(
        chunk_service.db,
        document_id,
    )


def test_get_chunk_success(chunk_service, sample_chunks) -> None:
    """Test getting a chunk by ID successfully."""
    chunk_id = 123
    expected_chunk = sample_chunks[0]
    chunk_service.chunk_repo.get = Mock(return_value=expected_chunk)

    result = chunk_service.get_chunk(chunk_id)

    assert result == expected_chunk
    chunk_service.chunk_repo.get.assert_called_once_with(chunk_service.db, id=chunk_id)


def test_get_chunk_not_found(chunk_service) -> None:
    """Test getting a non-existent chunk raises 404."""
    chunk_id = 123
    chunk_service.chunk_repo.get = Mock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        chunk_service.get_chunk(chunk_id)

    assert exc_info.value.status_code == 404
    assert str(chunk_id) in exc_info.value.detail


def test_delete_document_chunks_success(
    chunk_service, sample_document, sample_chunks
) -> None:
    """Test deleting all chunks for a document."""
    chunk_service.doc_repo.get = Mock(return_value=sample_document)
    chunk_service.chunk_repo.get_all_by_document = Mock(return_value=sample_chunks)
    chunk_service.chunk_repo.delete = Mock()

    chunk_service.delete_document_chunks(sample_document.id)

    chunk_service.doc_repo.get.assert_called_once_with(
        chunk_service.db,
        id=sample_document.id,
        lock=True,
    )
    chunk_service.chunk_repo.get_all_by_document.assert_called_once_with(
        chunk_service.db,
        sample_document.id,
    )
    assert chunk_service.chunk_repo.delete.call_count == 2


def test_delete_document_chunks_document_not_found(chunk_service) -> None:
    """Test deleting chunks for non-existent document raises 404."""
    document_id = 123
    chunk_service.doc_repo.get = Mock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        chunk_service.delete_document_chunks(document_id)

    assert exc_info.value.status_code == 404
    assert str(document_id) in exc_info.value.detail
