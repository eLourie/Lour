"""
app/services/embeddings/sparse.py

BM42 sparse embeddings via FastEmbed — served separately from Ollama.

Ollama does not expose a sparse-embedding endpoint, so the sparse leg of the
hybrid retriever runs through FastEmbed's ``SparseTextEmbedding``. BM42 uses
*different* encoders for documents and queries (attention-based term weighting
for passages, IDF-style weighting for queries), so this service exposes two
distinct methods.

FastEmbed is synchronous and CPU-bound (it also lazily downloads the model on
first use), so every call is offloaded to a worker thread to keep the event
loop responsive.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)


class SparseVector(BaseModel):
    """A sparse vector as parallel index / value arrays (Qdrant format)."""

    indices: list[int]
    values: list[float]

    @property
    def is_empty(self) -> bool:
        return len(self.indices) == 0


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    """Contract for sparse (lexical) embedding backends."""

    async def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode passages for indexing."""
        ...

    async def embed_query(self, text: str) -> SparseVector:
        """Encode a single query for search."""
        ...


def _to_sparse_vector(raw: Any) -> SparseVector:
    """Convert a FastEmbed SparseEmbedding (numpy arrays) to our model."""
    return SparseVector(
        indices=[int(i) for i in raw.indices],
        values=[float(v) for v in raw.values],
    )


class Bm42SparseEmbeddingService:
    """
    BM42 sparse embeddings using FastEmbed.

    The model is loaded lazily on first use (FastEmbed downloads it from the
    HuggingFace hub on demand). One instance is shared via lifespan.
    """

    def __init__(self, model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_model(self) -> Any:
        if self._model is None:
            async with self._lock:
                if self._model is None:  # double-checked under lock
                    logger.info("sparse_model_loading", model=self._model_name)
                    self._model = await asyncio.to_thread(self._load_model)
                    logger.info("sparse_model_ready", model=self._model_name)
        return self._model

    def _load_model(self) -> Any:
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(model_name=self._model_name)

    async def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        model = await self._ensure_model()

        def _run() -> list[SparseVector]:
            embeddings: Iterable[Any] = model.embed(texts)
            return [_to_sparse_vector(e) for e in embeddings]

        return await asyncio.to_thread(_run)

    async def embed_query(self, text: str) -> SparseVector:
        model = await self._ensure_model()

        def _run() -> SparseVector:
            embeddings: Iterable[Any] = model.query_embed([text])
            return [_to_sparse_vector(e) for e in embeddings][0]

        return await asyncio.to_thread(_run)
