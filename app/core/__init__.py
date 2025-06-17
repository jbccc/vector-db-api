"""Core library for vector database operations and document indexing."""

__version__ = "0.1.0"

from app.core.vector_db import get_vector_store

__all__ = [
    "get_vector_store",
]
