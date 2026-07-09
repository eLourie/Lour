"""
app/services/embeddings/base.py

EmbeddingProvider Protocol for dense vector generation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract for dense embedding backends."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one float vector per input text."""
        ...

    @property
    def dimensions(self) -> int:
        """Vector dimensionality (e.g. 1024 for bge-m3)."""
        ...
