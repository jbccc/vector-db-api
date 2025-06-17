from fastapi import APIRouter

from app.api.v1 import chunks, documents, libraries, vector_db

api_router = APIRouter(prefix="/v1")
api_router.include_router(libraries.router, prefix="/libraries", tags=["libraries"])
api_router.include_router(
    documents.router,
    prefix="/libraries/{library_id}/documents",
    tags=["documents"],
)
api_router.include_router(chunks.router, prefix="/chunks", tags=["chunks"])
api_router.include_router(vector_db.router, prefix="/vector-db", tags=["vector-db"])


@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "santé"}
