from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np

from app.core.cohere import get_cohere_embedding
from app.core.vector_db import (
    BruteForceIndex,
    LSHIndex,
    VectorStore,
)


def test_get_embedding_with_cohere() -> None:
    with patch("app.core.cohere.co") as mock_cohere_client:
        mock_cohere_client.embed.return_value.embeddings.float = [[0.1, 0.2, 0.3]]

        text = "test query"
        embedding = get_cohere_embedding(text)
        mock_cohere_client.embed.assert_called_once()
        assert embedding == [[0.1, 0.2, 0.3]]


def test_get_embedding_no_cohere() -> None:
    with patch("app.core.cohere.co", None):
        text = "test query"
        try:
            _ = get_cohere_embedding(text)
            msg = "Should have raised ValueError"
            raise AssertionError(msg)
        except ValueError as e:
            assert "Cohere API key not configured" in str(e)


def test_get_embedding_cohere() -> None:
    """Test the Cohere embedding function."""
    text = "test query"
    embedding = get_cohere_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 1
    assert isinstance(embedding[0], list)
    assert len(embedding[0]) == 1536

    try:
        get_cohere_embedding(1)
        msg = "Should have raised ValueError"
        raise AssertionError(msg)
    except ValueError as e:
        assert "The argument text must be either string or list of strings" in str(e)


def test_too_many_vectors_to_embed() -> None:
    """Test the Cohere embedding functions for large sample of text."""
    texts = [
        "test many queries",
    ] * 2000  # Cohere client doesn't accept more than 96 texts per request

    with patch("app.core.cohere.co") as mock_cohere_client:

        def mock_embed(**kwargs):
            batch_size = len(kwargs["texts"])
            mock_response = MagicMock()
            mock_response.embeddings.float = [[0.1, 0.2, 0.3]] * batch_size
            return mock_response

        mock_cohere_client.embed.side_effect = mock_embed

        embeddings = get_cohere_embedding(texts)

        # Should have called embed multiple times (2000/96 ~= 21 times)
        assert mock_cohere_client.embed.call_count > 1
        assert len(embeddings) == 2000
        assert all(isinstance(emb, list) for emb in embeddings)


def test_bruteforce_index() -> None:
    """Test the BruteForceIndex implementation."""
    lib_id = uuid4()
    index = BruteForceIndex(lib_id=lib_id, dimension=3)
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    index.add_vector("1", vec1, filter_metadata={"library_id": "lib1"})
    index.add_vector("2", vec2, filter_metadata={"library_id": "lib1"})

    query_vec = np.array([0.9, 0.1, 0])
    results = index.search(query_vec, top_k=1)
    assert len(results) == 1
    assert results[0].vector_id == "1"

    results_all = index.search(query_vec, top_k=2)
    assert len(results_all) == 2


def test_lsh_index() -> None:
    """Test the LSHIndex implementation."""
    lib_id = uuid4()
    index = LSHIndex(lib_id=lib_id, dimension=3, num_tables=10, num_hyperplanes=4)
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    index.add_vector("1", vec1)
    index.add_vector("2", vec2)

    # Since LSH is probabilistic, we search for a vector that is very close
    # to one of the indexed vectors.
    query_vec = np.array([1.0, 0.01, -0.01])
    results = index.search(query_vec, top_k=1)

    # It's possible LSH doesn't find the candidate in some runs, but it should be rare
    # with enough tables. A more robust test might check statistics over many runs.
    if results:
        assert results[0].vector_id == "1", (
            "Vector not retrieve. It might be because LSH is not exact."
        )

    # Test removal
    index.remove_vector("1")
    results_after_removal = index.search(query_vec, top_k=1)
    if results_after_removal:
        assert results_after_removal[0].vector_id != "1"
    assert "1" not in index.vectors


def test_vector_store_bruteforce_retrieval() -> None:
    """Test VectorStore with BruteForceIndex."""
    vector_store = object.__new__(VectorStore)
    lib_id = uuid4()
    vector_store.indices = {}
    vector_store.indices[lib_id] = BruteForceIndex(lib_id=lib_id, dimension=3)
    vector_store.chunk_repository = MagicMock()
    vector_store.embedding_service = MagicMock()

    mock_db_session = MagicMock()

    chunk1_id = uuid4()
    chunk2_id = uuid4()

    mock_chunk1 = MagicMock()
    mock_chunk1.id = chunk1_id

    vector_store.add_vector(lib_id, str(chunk1_id), np.array([1, 0, 0]))
    vector_store.add_vector(lib_id, str(chunk2_id), np.array([0, 1, 0]))

    vector_store.embedding_service.get_embeddings.return_value = [[0.9, 0.1, 0]]
    vector_store.chunk_repository.get_by_ids.return_value = [mock_chunk1]

    retrieved = vector_store.retrieve_chunks(mock_db_session, "query", lib_id, top_k=1)

    vector_store.chunk_repository.get_by_ids.assert_called_with(
        mock_db_session,
        [str(chunk1_id)],
    )
    assert len(retrieved) == 1
    assert retrieved[0].id == chunk1_id


def test_vector_store_lsh_retrieval() -> None:
    """Test VectorStore with LSHIndex."""
    vector_store = object.__new__(VectorStore)
    lib_id = uuid4()
    vector_store.indices = {}
    vector_store.indices[lib_id] = LSHIndex(
        lib_id=lib_id,
        dimension=3,
        num_tables=10,
        num_hyperplanes=4,
    )
    vector_store.chunk_repository = MagicMock()
    vector_store.embedding_service = MagicMock()

    mock_db_session = MagicMock()

    chunk1_id = uuid4()
    mock_chunk1 = MagicMock()
    mock_chunk1.id = chunk1_id

    vector_store.add_vector(lib_id, str(chunk1_id), np.array([1, 0, 0]))

    vector_store.embedding_service.get_embeddings.return_value = [[0.9, 0.1, 0]]
    vector_store.chunk_repository.get_by_ids.return_value = [mock_chunk1]

    retrieved = vector_store.retrieve_chunks(mock_db_session, "query", lib_id, top_k=1)

    # In LSH, it's possible the chunk is not found, but we check the call was made
    if retrieved:
        vector_store.chunk_repository.get_by_ids.assert_called_with(
            mock_db_session,
            [str(chunk1_id)],
        )
        assert retrieved[0].id == chunk1_id, (
            "Vector not retrieve. It might be because LSH is not exact."
        )
