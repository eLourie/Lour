"""
app/services/reranker/base.py

Reranker Protocol and mode enum.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """
    Contract for cross-encoder rerankers.

    Returns scores in the same order as *documents*.
    Higher score = more relevant to the query.
    """

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score each document for relevance to *query*."""
        ...

    @property
    def available(self) -> bool:
        """Return True if the reranker is loaded and ready."""
        ...
