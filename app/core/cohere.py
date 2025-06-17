"""Cohere API utilities."""

from __future__ import annotations

import contextlib

import cohere
from fastapi import HTTPException

from app.config import settings

co = None
with contextlib.suppress(Exception):
    co = cohere.ClientV2(settings.COHERE_API_KEY)


def get_cohere_embedding(text: str | list[str]) -> list[float]:
    """Generate embedding using Cohere API."""
    if isinstance(text, str):
        text = [text]
    elif not isinstance(text, list) or not isinstance(text[0], str):
        msg = f"The argument text must be either string or list of strings, got {type(text)}"
        raise ValueError(
            msg,
        )
    if not co:
        msg = "Cohere API key not configured"
        raise ValueError(msg)

    max_batch_size = 96  # Cohere API has a limit of 96 texts per request
    all_embeddings = []

    try:
        for i in range(0, len(text), max_batch_size):
            batch = text[i : i + max_batch_size]
            response = co.embed(
                texts=batch,
                model=settings.COHERE_MODEL,
                input_type=settings.COHERE_INPUT_TYPE,
                embedding_types=["float"],
            )
            all_embeddings.extend(response.embeddings.float)

        return all_embeddings

    except cohere.errors.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid Cohere API key")
    except cohere.errors.TooManyRequestsError:
        raise HTTPException(status_code=429, detail="Cohere API rate limit exceeded")
    except cohere.errors.BadRequestError as e:
        raise HTTPException(status_code=400, detail=f"Bad request to Cohere API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere API error: {e}")
