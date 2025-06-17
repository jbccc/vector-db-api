from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.core.cohere import get_cohere_embedding
from app.db import ChunkRepository
from app.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using external APIs."""

    def get_embeddings(self, text: list[str] | str) -> list[list[float]]:
        """Generate embeddings for given text(s)."""
        try:
            return get_cohere_embedding(text)
        except Exception as e:
            logger.exception(f"Failed to generate embeddings: {e}")
            msg = f"Embedding generation failed: {e}"
            raise RuntimeError(msg) from e


class VectorIndex(ABC):
    """Abstract base class for vector indexes."""

    def __init__(self, lib_id: UUID, dimension: int) -> None:
        """Initialize the VectorIndex with library ID and dimension."""
        self.lib_id = lib_id
        if dimension <= 0:
            msg = f"Dimension must be positive, got {dimension}"
            raise ValueError(msg)
        self.dimension = dimension
        self.vectors: dict[str, np.ndarray] = {}

    def add_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        filter_metadata: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """Add a single vector to the index."""
        self._validate_vector(vector_id, vector)
        self.vectors[vector_id] = vector.copy()
        logger.debug(f"Added vector {vector_id} to index")

    def add_vectors_batch(
        self,
        vectors: dict[str, np.ndarray],
        filter_metadata: dict[str, dict[str, Any]] | None = None,  # noqa: ARG002
    ) -> None:
        """Add multiple vectors to the index efficiently."""
        if not vectors:
            return

        for vector_id, vector in vectors.items():
            self._validate_vector(vector_id, vector)

        for vector_id, vector in vectors.items():
            self.vectors[vector_id] = vector.copy()

        logger.info(f"Added {len(vectors)} vectors to index in batch")

    def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the index. Returns True if removed, False if not found."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            logger.debug(f"Removed vector {vector_id} from index")
            return True
        return False

    def remove_vectors_batch(self, vector_ids: list[str]) -> int:
        """Remove multiple vectors from the index. Returns count of removed vectors."""
        removed_count = 0
        for vector_id in vector_ids:
            if self.remove_vector(vector_id):
                removed_count += 1
        logger.info(f"Removed {removed_count}/{len(vector_ids)} vectors from index")
        return removed_count

    def _validate_vector(self, vector_id: str, vector: np.ndarray) -> None:
        """Validate vector dimensions and ID."""
        if not vector_id or not isinstance(vector_id, str):
            msg = "Vector ID must be a non-empty string"
            raise ValueError(msg)

        if not isinstance(vector, np.ndarray):
            msg = "Vector must be a numpy array"
            raise TypeError(msg)

        if vector.ndim != 1:
            msg = f"Vector must be 1-dimensional, got {vector.ndim}D"
            raise ValueError(msg)

        if vector.shape[0] != self.dimension:
            msg = f"Vector has dimension {vector.shape[0]} but index expects {self.dimension}"
            raise ValueError(
                msg,
            )

        if not np.isfinite(vector).all():
            msg = "Vector contains non-finite values"
            raise ValueError(msg)

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """Search for the most similar vectors."""

    def get_vector_count(self) -> int:
        """Get the number of vectors in the index."""
        return len(self.vectors)


class BruteForceIndex(VectorIndex):
    """Exact k-nearest neighbors search using brute-force cosine similarity."""

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """Search for the most similar vectors using brute-force cosine similarity."""
        if top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)

        self._validate_vector("query", query_vector)

        if not self.vectors:
            return []

        vector_ids = list(self.vectors.keys())
        vectors_matrix = np.vstack(list(self.vectors.values()))

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            logger.warning("Query vector has zero norm")
            return []

        vectors_norm = np.linalg.norm(vectors_matrix, axis=1)
        valid_mask = vectors_norm > 0

        if not np.any(valid_mask):
            logger.warning("No valid vectors found (all have zero norm)")
            return []

        similarities = np.zeros(len(vector_ids))
        similarities[valid_mask] = np.dot(vectors_matrix[valid_mask], query_vector) / (
            query_norm * vectors_norm[valid_mask]
        )

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return []

        sorted_indices = np.argsort(similarities[valid_mask])[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            original_idx = valid_indices[idx]
            results.append(
                SearchResult(
                    vector_id=vector_ids[original_idx],
                    similarity_score=float(similarities[original_idx]),
                ),
            )

        return results


class LSHIndex(VectorIndex):
    """Approximate nearest neighbor index using Locality-Sensitive Hashing."""

    def __init__(
        self,
        lib_id: UUID,
        dimension: int,
        num_tables: int = 10,
        num_hyperplanes: int = 8,
        random_seed: int = 42,
    ) -> None:
        """Initialize the LSHIndex with configuration parameters."""
        super().__init__(lib_id, dimension)
        if num_tables <= 0 or num_hyperplanes <= 0:
            msg = "num_tables and num_hyperplanes must be positive"
            raise ValueError(msg)

        self.num_tables = num_tables
        self.num_hyperplanes = num_hyperplanes
        self.random_seed = random_seed

        self.hash_tables: list[defaultdict[str, list[str]]] = [
            defaultdict(list) for _ in range(num_tables)
        ]

        np.random.seed(random_seed)
        self.hyperplanes: list[np.ndarray] = [
            np.random.randn(dimension, num_hyperplanes) for _ in range(num_tables)
        ]
        np.random.seed()

    def _hash(self, vector: np.ndarray, table_index: int) -> str:
        """Compute LSH hash for a vector."""
        projections = np.dot(vector, self.hyperplanes[table_index])
        return "".join(["1" if p > 0 else "0" for p in projections])

    def add_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        filter_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the LSH index."""
        super().add_vector(vector_id, vector, filter_metadata)
        for i in range(self.num_tables):
            h = self._hash(vector, i)
            self.hash_tables[i][h].append(vector_id)

    def add_vectors_batch(
        self,
        vectors: dict[str, np.ndarray],
        filter_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple vectors to the LSH index efficiently."""
        super().add_vectors_batch(vectors, filter_metadata)

        for vector_id, vector in vectors.items():
            for i in range(self.num_tables):
                hash_key = self._hash(vector, i)
                self.hash_tables[i][hash_key].append(vector_id)

    def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the LSH index."""
        if vector_id not in self.vectors:
            return False

        vector_to_remove = self.vectors[vector_id]

        for i in range(self.num_tables):
            hash_key = self._hash(vector_to_remove, i)
            if (
                hash_key in self.hash_tables[i]
                and vector_id in self.hash_tables[i][hash_key]
            ):
                self.hash_tables[i][hash_key].remove(vector_id)
                if not self.hash_tables[i][hash_key]:
                    del self.hash_tables[i][hash_key]

        super().remove_vector(vector_id)
        return True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """Search for the closest vectors in the index."""
        if top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)

        self._validate_vector("query", query_vector)

        candidate_ids = set()
        for i in range(self.num_tables):
            h = self._hash(query_vector, i)
            candidate_ids.update(self.hash_tables[i][h])

        if not candidate_ids:
            logger.debug("No candidates found in LSH search")
            return []

        candidate_vectors = {vid: self.vectors[vid] for vid in candidate_ids}

        if not candidate_vectors:
            return []

        vector_ids = list(candidate_vectors.keys())
        vectors_matrix = np.vstack(list(candidate_vectors.values()))

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []

        vectors_norm = np.linalg.norm(vectors_matrix, axis=1)
        valid_mask = vectors_norm > 0

        if not np.any(valid_mask):
            return []

        similarities = np.zeros(len(vector_ids))
        similarities[valid_mask] = np.dot(vectors_matrix[valid_mask], query_vector) / (
            query_norm * vectors_norm[valid_mask]
        )

        valid_indices = np.where(valid_mask)[0]
        sorted_indices = np.argsort(similarities[valid_mask])[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            original_idx = valid_indices[idx]
            results.append(
                SearchResult(
                    vector_id=vector_ids[original_idx],
                    similarity_score=float(similarities[original_idx]),
                ),
            )

        return results


class VectorIndexFactory:
    """Factory for creating vector indexes."""

    @staticmethod
    def create_index(
        index_type: str,
        lib_id: UUID,
        dimension: int,
        **kwargs,
    ) -> VectorIndex:
        """Create a vector index of the specified type."""
        if index_type == "bruteforce":
            return BruteForceIndex(lib_id=lib_id, dimension=dimension)
        if index_type == "lsh":
            return LSHIndex(
                lib_id=lib_id,
                dimension=dimension,
                num_tables=kwargs.get("num_tables", 10),
                num_hyperplanes=kwargs.get("num_hyperplanes", 8),
                random_seed=kwargs.get("random_seed", 42),
            )
        msg = f"Unsupported vector index type: {index_type}"
        raise ValueError(msg)


class VectorStore:
    """Vector store for managing embeddings and similarity search across multiple libraries."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chunk_repository: ChunkRepository,
    ) -> None:
        """Initialize the VectorStore with embedding service and chunk repository."""
        self.indices: dict[UUID, VectorIndex] = {}
        self.embedding_service = embedding_service
        self.chunk_repository = chunk_repository

    def create_index(
        self,
        lib_id: UUID,
        index_type: str | None = None,
        **kwargs,
    ) -> VectorIndex:
        """Create a new vector index for the given library ID."""
        if lib_id in self.indices:
            logger.warning(
                f"Index for library {lib_id} already exists, returning existing index",
            )
            return self.indices[lib_id]

        index_type = index_type or settings.VECTOR_INDEX_TYPE
        dimension = settings.VECTOR_DIMENSION

        index = VectorIndexFactory.create_index(
            index_type=index_type,
            lib_id=lib_id,
            dimension=dimension,
            num_tables=kwargs.get(
                "num_tables",
                getattr(settings, "LSH_NUM_TABLES", 10),
            ),
            num_hyperplanes=kwargs.get(
                "num_hyperplanes",
                getattr(settings, "LSH_NUM_HYPERPLANES", 8),
            ),
            random_seed=kwargs.get(
                "random_seed",
                getattr(settings, "LSH_RANDOM_SEED", 42),
            ),
        )

        self.indices[lib_id] = index
        logger.info(f"Created {index_type} index for library {lib_id}")
        return index

    def get_index(self, lib_id: UUID) -> VectorIndex | None:
        """Retrieve the vector index for the given library ID."""
        return self.indices.get(lib_id)

    def has_index(self, lib_id: UUID) -> bool:
        """Check if an index exists for the given library ID."""
        return lib_id in self.indices

    def get_or_create_index(
        self,
        lib_id: UUID,
        index_type: str | None = None,
        **kwargs,
    ) -> VectorIndex:
        """Get existing index or create a new one for the given library ID."""
        if self.has_index(lib_id):
            return self.get_index(lib_id)
        return self.create_index(lib_id, index_type, **kwargs)

    def remove_index(self, lib_id: UUID) -> bool:
        """Remove the index for the given library ID."""
        if lib_id in self.indices:
            del self.indices[lib_id]
            logger.info(f"Removed index for library {lib_id}")
            return True
        return False

    def add_text(self, lib_id: UUID, vector_id: str, text: str) -> None:
        """Add text to the vector store by generating embeddings."""
        index = self.get_or_create_index(lib_id)
        try:
            embeddings = self.embedding_service.get_embeddings(text)
            vector = np.array(embeddings[0])
            index.add_vector(vector_id, vector)
        except Exception as e:
            logger.exception(
                f"Failed to add text for vector_id {vector_id} in library {lib_id}: {e}",
            )
            raise

    def add_texts_batch(self, lib_id: UUID, texts: dict[str, str]) -> None:
        """Add multiple texts to the vector store efficiently."""
        if not texts:
            return

        index = self.get_or_create_index(lib_id)
        try:
            text_list = list(texts.values())
            embeddings = self.embedding_service.get_embeddings(text_list)

            vectors = {}
            for i, vector_id in enumerate(texts.keys()):
                vectors[vector_id] = np.array(embeddings[i])

            index.add_vectors_batch(vectors)
        except Exception as e:
            logger.exception(f"Failed to add texts in batch for library {lib_id}: {e}")
            raise

    def add_vector(self, lib_id: UUID, vector_id: str, vector: np.ndarray) -> None:
        """Add a pre-computed vector to the store."""
        index = self.get_or_create_index(lib_id)
        index.add_vector(vector_id, vector)

    def add_vectors_batch(self, lib_id: UUID, vectors: dict[str, np.ndarray]) -> None:
        """Add multiple pre-computed vectors to the store."""
        index = self.get_or_create_index(lib_id)
        index.add_vectors_batch(vectors)

    def remove_vector(self, lib_id: UUID, vector_id: str) -> bool:
        """Remove a vector from the store."""
        index = self.get_index(lib_id)
        if index:
            return index.remove_vector(vector_id)
        return False

    def remove_vectors_batch(self, lib_id: UUID, vector_ids: list[str]) -> int:
        """Remove multiple vectors from the store."""
        index = self.get_index(lib_id)
        if index:
            return index.remove_vectors_batch(vector_ids)
        return 0

    def retrieve_chunks(
        self,
        db: Session,
        query: str,
        library_id: UUID,
        top_k: int,
    ) -> list[Chunk]:
        """Retrieve chunks from database based on similarity search."""
        index = self.get_index(library_id)
        if not index:
            logger.warning(f"No index found for library {library_id}")
            return []

        try:
            query_embeddings = self.embedding_service.get_embeddings(query)
            query_vector = np.array(query_embeddings[0])
            search_results = index.search(query_vector, top_k)

            if not search_results:
                return []

            chunk_ids = [result.vector_id for result in search_results]

            chunks = self.chunk_repository.get_by_ids(db, chunk_ids)
            chunk_lookup = {str(chunk.id): chunk for chunk in chunks}
            ordered_chunks = [
                chunk_lookup[chunk_id]
                for chunk_id in chunk_ids
                if chunk_id in chunk_lookup
            ]

            logger.info(
                f"Retrieved {len(ordered_chunks)} chunks for query in library {library_id}",
            )
            return ordered_chunks

        except Exception as e:
            logger.exception(f"Failed to retrieve chunks for library {library_id}: {e}")
            raise

    def get_vector_count(self, lib_id: UUID) -> int:
        """Get the number of vectors in the store for a given library."""
        index = self.get_index(lib_id)
        if index:
            return index.get_vector_count()
        return 0

    def get_library_ids(self) -> list[UUID]:
        """Get all library IDs that have indices."""
        return list(self.indices.keys())

    def delete(self, document_id: str) -> None:
        """Delete all vectors associated with a document across all libraries."""
        for lib_id, index in self.indices.items():
            index.remove_vector(document_id)
            logger.debug(f"Removed document {document_id} from library {lib_id}")


def create_vector_store() -> VectorStore:
    """Create a configured vector store."""
    embedding_service = EmbeddingService()
    chunk_repository = ChunkRepository()
    return VectorStore(embedding_service, chunk_repository)


_vector_store: VectorStore | None = None


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    global _vector_store

    if os.getenv("PYTEST_CURRENT_TEST"):
        return create_vector_store()

    if _vector_store is None:
        _vector_store = create_vector_store()

    return _vector_store
