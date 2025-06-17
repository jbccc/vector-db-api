"""RAG API Application Package.

This package contains a FastAPI application for document management and search
using Retrieval-Augmented Generation (RAG) capabilities.
"""

__version__ = "0.1.0"
__author__ = "Jean-Baptiste Conan"
__description__ = "StackAI Takehome assignment - coding a vector search engine from scratch using FastAPI and Pydantic."

from app.main import app

__all__ = ["app"]
