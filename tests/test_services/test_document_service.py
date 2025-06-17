from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Document, ProcessingStatus
from app.services import DocumentService


@pytest.fixture
def mock_db():
    return Mock(spec=Session)


@pytest.fixture
def document_service(mock_db):
    return DocumentService(mock_db)


@pytest.fixture
def sample_document():
    return Document(
        id=uuid4(),
        title="Test Document",
        content="Test content",
        library_id=uuid4(),
        processing_status=ProcessingStatus.NOT_INDEXED,
    )


def test_get_document_success(document_service, sample_document) -> None:
    """Test getting a document by ID successfully."""
    mock_db_doc = Mock()
    mock_db_doc.to_dict.return_value = {
        "id": sample_document.id,
        "title": sample_document.title,
        "content": sample_document.content,
        "library_id": sample_document.library_id,
        "processing_status": sample_document.processing_status,
    }
    document_service.doc_repo.get = Mock(return_value=mock_db_doc)

    result = document_service.get_document(sample_document.id)

    assert result == mock_db_doc
    document_service.doc_repo.get.assert_called_once_with(
        document_service.db,
        id=sample_document.id,
        lock=False,
    )


def test_get_document_with_lock(document_service, sample_document) -> None:
    """Test getting a document with lock enabled."""
    mock_db_doc = Mock()
    document_service.doc_repo.get = Mock(return_value=mock_db_doc)

    result = document_service.get_document(sample_document.id, lock=True)

    assert result == mock_db_doc
    document_service.doc_repo.get.assert_called_once_with(
        document_service.db,
        id=sample_document.id,
        lock=True,
    )


def test_get_document_not_found(document_service) -> None:
    """Test getting a non-existent document raises 404."""
    document_id = uuid4()
    document_service.doc_repo.get = Mock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        document_service.get_document(document_id)

    assert exc_info.value.status_code == 404
    assert str(document_id) in exc_info.value.detail


def test_mark_document_as_indexing(document_service, sample_document) -> None:
    """Test marking document as indexing."""
    mock_db_doc = Mock()
    mock_db_doc.to_dict.return_value = {
        "id": sample_document.id,
        "title": sample_document.title,
        "content": sample_document.content,
        "library_id": sample_document.library_id,
        "processing_status": ProcessingStatus.NOT_INDEXED,
        "processed_at": None,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    document_service.doc_repo.get = Mock(return_value=mock_db_doc)
    document_service.doc_repo.update = Mock()

    document_service.mark_document_as_indexing(sample_document.id)

    document_service.doc_repo.get.assert_called_once_with(
        document_service.db,
        id=sample_document.id,
        lock=True,
    )
    document_service.doc_repo.update.assert_called_once()
    call_args = document_service.doc_repo.update.call_args
    assert call_args[1]["id"] == sample_document.id
    update_data = call_args[1]["obj_in"]
    assert update_data["processing_status"] == ProcessingStatus.INDEXING


def test_mark_document_as_indexed(document_service, sample_document) -> None:
    """Test marking document as indexed includes processed_at timestamp."""
    mock_db_doc = Mock()
    mock_db_doc.to_dict.return_value = {
        "id": sample_document.id,
        "title": sample_document.title,
        "content": sample_document.content,
        "library_id": sample_document.library_id,
        "processing_status": ProcessingStatus.INDEXING,
        "processed_at": None,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    document_service.doc_repo.get = Mock(return_value=mock_db_doc)
    document_service.doc_repo.update = Mock()

    document_service.mark_document_as_indexed(sample_document.id)

    # Verify the update was called with both status and processed_at
    call_args = document_service.doc_repo.update.call_args
    assert call_args[1]["id"] == sample_document.id
    update_data = call_args[1]["obj_in"]
    assert update_data["processing_status"] == ProcessingStatus.INDEXED
    assert "processed_at" in update_data
    assert isinstance(update_data["processed_at"], datetime)


def test_mark_document_as_failed(document_service, sample_document) -> None:
    """Test marking document as failed."""
    mock_db_doc = Mock()
    mock_db_doc.to_dict.return_value = {
        "id": sample_document.id,
        "title": sample_document.title,
        "content": sample_document.content,
        "library_id": sample_document.library_id,
        "processing_status": ProcessingStatus.INDEXING,
        "processed_at": None,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    document_service.doc_repo.get = Mock(return_value=mock_db_doc)
    document_service.doc_repo.update = Mock()

    document_service.mark_document_as_failed(sample_document.id)

    call_args = document_service.doc_repo.update.call_args
    assert call_args[1]["id"] == sample_document.id
    update_data = call_args[1]["obj_in"]
    assert update_data["processing_status"] == ProcessingStatus.FAILED
    assert update_data["processed_at"] is None


def test_mark_document_as_not_indexed(document_service, sample_document) -> None:
    """Test marking document as not indexed."""
    mock_db_doc = Mock()
    mock_db_doc.to_dict.return_value = {
        "id": sample_document.id,
        "title": sample_document.title,
        "content": sample_document.content,
        "library_id": sample_document.library_id,
        "processing_status": ProcessingStatus.FAILED,
        "processed_at": None,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    document_service.doc_repo.get = Mock(return_value=mock_db_doc)
    document_service.doc_repo.update = Mock()

    document_service.mark_document_as_not_indexed(sample_document.id)

    call_args = document_service.doc_repo.update.call_args
    assert call_args[1]["id"] == sample_document.id
    update_data = call_args[1]["obj_in"]
    assert update_data["processing_status"] == ProcessingStatus.NOT_INDEXED
    assert update_data["processed_at"] is None


def test_update_document_success(document_service, sample_document) -> None:
    """Test updating document data successfully."""
    mock_db_doc = Mock()
    document_service.doc_repo.get = Mock(return_value=mock_db_doc)

    updated_doc = Mock()
    document_service.doc_repo.update = Mock(return_value=updated_doc)

    update_data = {"title": "Updated Title", "content": "Updated content"}

    result = document_service.update_document(sample_document.id, update_data)

    assert result == updated_doc
    document_service.doc_repo.get.assert_called_once_with(
        document_service.db,
        id=sample_document.id,
        lock=True,
    )
    document_service.doc_repo.update.assert_called_once_with(
        document_service.db,
        id=sample_document.id,
        obj_in=update_data,
    )


def test_update_document_integrity_error(document_service, sample_document) -> None:
    """Test updating document with integrity constraint violation."""
    from sqlalchemy.exc import IntegrityError

    mock_db_doc = Mock()
    document_service.doc_repo.get = Mock(return_value=mock_db_doc)
    document_service.doc_repo.update = Mock(
        side_effect=IntegrityError(None, None, None),
    )
    document_service.db.rollback = Mock()

    update_data = {"title": "Duplicate Title"}

    with pytest.raises(HTTPException) as exc_info:
        document_service.update_document(sample_document.id, update_data)

    assert exc_info.value.status_code == 409
    assert "already exists" in exc_info.value.detail
    document_service.db.rollback.assert_called_once()


def test_update_document_not_found(document_service) -> None:
    """Test updating non-existent document raises 404."""
    document_id = uuid4()
    document_service.doc_repo.get = Mock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        document_service.update_document(document_id, {"title": "New Title"})

    assert exc_info.value.status_code == 404
    assert str(document_id) in exc_info.value.detail
