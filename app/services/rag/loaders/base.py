"""
app/services/rag/loaders/base.py

Loader contract and the LoadedDocument value object.

A loader turns a *source* (file path or URL) into a :class:`LoadedDocument`:
normalised text plus metadata. Some loaders (notably code) already know the
natural chunk boundaries and expose them via ``segments``; the ingestion
pipeline uses those verbatim and only falls back to the semantic chunker when
``segments`` is ``None``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """Normalised output of a loader, ready for the ingestion pipeline."""

    content: str
    source_uri: str
    doc_type: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Pre-computed chunk boundaries (e.g. code symbols). None → run the chunker.
    segments: list[str] | None = None


@runtime_checkable
class Loader(Protocol):
    """Contract for source loaders."""

    doc_type: str

    def supports(self, source: str) -> bool:
        """Return True if this loader can handle *source*."""
        ...

    async def load(self, source: str) -> LoadedDocument:
        """Load and normalise *source*."""
        ...
