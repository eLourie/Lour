"""
app/services/embeddings/bge_m3.py

Dense embedding service using bge-m3 via Ollama (1024 dimensions).
"""

from __future__ import annotations

from app.infra.clients.ollama import OllamaClient


class BgeM3EmbeddingService:
    """
    Generates 1024-dim dense embeddings using bge-m3 through Ollama.

    One instance is shared via lifespan (singleton).
    Caching is applied externally via :class:`~app.services.embeddings.cache.EmbeddingCache`.
    """

    DIMENSIONS = 1024

    def __init__(self, client: OllamaClient, model: str = "bge-m3") -> None:
        self._client = client
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._client.embed(model=self._model, input=texts)

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS