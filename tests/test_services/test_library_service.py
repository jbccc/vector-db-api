from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Document, Library
from app.services import LibraryService


@pytest.fixture
def mock_db():
    return Mock(spec=Session)


@pytest.fixture
def library_service(mock_db):
    return LibraryService(mock_db)


@pytest.fixture
def sample_library_data():
    return {"name": "Test Library", "description": "A test library"}


@pytest.fixture
def sample_library():
    return Library(id=uuid4(), name="Test Library", description="A test library")


def test_create_library_success(
    library_service, sample_library_data, sample_library
) -> None:
    """Test creating a new library successfully."""
    library_service.repo.create = Mock(return_value=sample_library)

    result = library_service.create_library(sample_library_data)

    assert result == sample_library
    library_service.repo.create.assert_called_once_with(
        library_service.db,
        obj_in=sample_library_data,
    )


def test_create_library_conflict(library_service, sample_library_data) -> None:
    """Test creating a library with duplicate name raises conflict."""
    from sqlalchemy.exc import IntegrityError

    library_service.repo.create = Mock(side_effect=IntegrityError(None, None, None))

    with pytest.raises(HTTPException) as exc_info:
        library_service.create_library(sample_library_data)

    assert exc_info.value.status_code == 409
    assert "already exists" in exc_info.value.detail


def test_list_libraries_no_search(library_service) -> None:
    """Test listing all libraries without search."""
    expected_libraries = [
        Library(id=uuid4(), name="Library 1", description="Desc 1"),
        Library(id=uuid4(), name="Library 2", description="Desc 2"),
    ]
    library_service.repo.get_all = Mock(return_value=expected_libraries)

    result = library_service.list_libraries()

    assert result == expected_libraries
    library_service.repo.get_all.assert_called_once_with(library_service.db)


def test_list_libraries_with_search(library_service) -> None:
    """Test listing libraries with search query."""
    search_query = "test"
    expected_libraries = [Library(id=uuid4(), name="Test Library", description="Desc")]
    library_service.repo.search_by_name = Mock(return_value=expected_libraries)

    result = library_service.list_libraries(search=search_query)

    assert result == expected_libraries
    library_service.repo.search_by_name.assert_called_once_with(
        library_service.db,
        query=search_query,
    )


def test_get_library_success(library_service, sample_library) -> None:
    """Test getting a library by valid ID."""
    library_service.repo.get = Mock(return_value=sample_library)

    result = library_service.get_library(sample_library.id)

    assert result == sample_library
    library_service.repo.get.assert_called_once_with(
        library_service.db,
        id=sample_library.id,
        lock=False,
    )


def test_get_library_not_found(library_service) -> None:
    """Test getting a library with invalid ID raises 404."""
    library_id = uuid4()
    library_service.repo.get = Mock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        library_service.get_library(library_id)

    assert exc_info.value.status_code == 404
    assert str(library_id) in exc_info.value.detail


def test_update_library_success(library_service, sample_library) -> None:
    """Test updating an existing library."""
    update_data = {"name": "Updated Library", "description": "Updated description"}
    updated_library = Library(id=sample_library.id, **update_data)

    library_service.repo.get = Mock(return_value=sample_library)
    library_service.repo.get_by_name = Mock(return_value=None)
    library_service.repo.update = Mock(return_value=updated_library)

    result = library_service.update_library(sample_library.id, update_data)

    assert result == updated_library
    library_service.repo.update.assert_called_once_with(
        library_service.db,
        id=sample_library.id,
        obj_in=update_data,
    )


def test_update_library_name_conflict(library_service, sample_library) -> None:
    """Test updating library name to existing name raises conflict."""
    update_data = {"name": "Existing Library"}
    existing_library = Library(id=uuid4(), name="Existing Library", description="Desc")

    library_service.repo.get = Mock(return_value=sample_library)
    library_service.repo.get_by_name = Mock(return_value=existing_library)

    with pytest.raises(HTTPException) as exc_info:
        library_service.update_library(sample_library.id, update_data)

    assert exc_info.value.status_code == 409
    assert "already exists" in exc_info.value.detail


def test_delete_library_success(library_service, sample_library) -> None:
    """Test deleting a library."""
    library_service.repo.get = Mock(return_value=sample_library)
    library_service.repo.delete = Mock()

    library_service.delete_library(sample_library.id)

    library_service.repo.delete.assert_called_once_with(
        library_service.db,
        id=sample_library.id,
    )


def test_delete_library_not_found(library_service) -> None:
    """Test deleting non-existent library raises 404."""
    library_id = uuid4()
    library_service.repo.get = Mock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        library_service.delete_library(library_id)

    assert exc_info.value.status_code == 404


def test_list_library_documents(library_service, sample_library) -> None:
    """Test listing documents in a library."""
    mock_doc_dict = {
        "id": uuid4(),
        "title": "Test Document",
        "content": "Test content",
        "library_id": sample_library.id,
    }
    mock_doc = Mock()
    mock_doc.to_dict.return_value = mock_doc_dict

    library_service.repo.get = Mock(return_value=sample_library)
    library_service.repo.get_documents = Mock(return_value=[mock_doc])

    result = library_service.list_library_documents(sample_library.id)

    assert len(result) == 1
    assert isinstance(result[0], Document)
    library_service.repo.get_documents.assert_called_once_with(
        library_service.db,
        sample_library.id,
    )
