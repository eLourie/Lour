"""
app/services/rag/chunking.py

Chunking strategies for ingestion.

Two chunkers:
  - RecursiveCharacterChunker — deterministic, dependency-free fallback that
    splits on a separator hierarchy and merges pieces up to a target size with
    overlap. Always available (no embeddings, no I/O).
  - SemanticChunker — splits on *meaning*: it embeds sentences and cuts where
    adjacent sentences become dissimilar (a distance percentile breakpoint).
    Falls back to the recursive chunker when embeddings are unavailable or the
    text is too short to benefit.

Semantic chunking is more expensive (one embedding call per document), so the
ingestion pipeline uses it for prose and lets the code loader bypass it with
symbol-boundary segments.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.services.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


class RecursiveCharacterChunker:
    """Split text on a separator hierarchy, merging up to ``chunk_size``."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: tuple[str, ...] = _DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        pieces = self._split_recursive(text, self._separators)
        return self._merge_with_overlap(pieces)

    def _split_recursive(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []
        if not separators:
            return [text[i : i + self._chunk_size] for i in range(0, len(text), self._chunk_size)]

        sep, *rest = separators
        parts = list(text) if sep == "" else text.split(sep)
        out: list[str] = []
        for part in parts:
            if len(part) <= self._chunk_size:
                if part.strip():
                    out.append(part)
            else:
                out.extend(self._split_recursive(part, tuple(rest)))
        return out

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self._chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if self._chunk_overlap and chunks:
                tail = chunks[-1][-self._chunk_overlap :]
                current = f"{tail} {piece}".strip()
                if len(current) > self._chunk_size:
                    current = piece
            else:
                current = piece
        if current:
            chunks.append(current)
        return chunks


class SemanticChunker:
    """
    Sentence-similarity chunker with a recursive character fallback.

    The embedder is optional so the chunker degrades gracefully: without it (or
    on any embedding failure) it delegates to :class:`RecursiveCharacterChunker`.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        *,
        max_chunk_chars: int = 2000,
        min_sentences: int = 4,
        breakpoint_percentile: float = 90.0,
        fallback: RecursiveCharacterChunker | None = None,
    ) -> None:
        self._embedder = embedder
        self._max_chunk_chars = max_chunk_chars
        self._min_sentences = min_sentences
        self._breakpoint_percentile = breakpoint_percentile
        self._fallback = fallback or RecursiveCharacterChunker(
            chunk_size=max_chunk_chars,
            chunk_overlap=min(150, max_chunk_chars // 4),
        )

    async def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if self._embedder is None or len(sentences) < self._min_sentences:
            return self._fallback.split(text)

        try:
            vectors = await self._embedder.embed(sentences)
        except Exception as exc:
            logger.warning("semantic_chunk_embed_failed", error=str(exc))
            return self._fallback.split(text)

        distances = self._consecutive_distances(vectors)
        if not distances:
            return self._fallback.split(text)

        threshold = float(np.percentile(distances, self._breakpoint_percentile))
        chunks = self._group_by_breakpoints(sentences, distances, threshold)
        return self._enforce_max_size(chunks)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

    @staticmethod
    def _consecutive_distances(vectors: list[list[float]]) -> list[float]:
        if len(vectors) < 2:
            return []
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        unit = arr / norms
        cos = np.sum(unit[:-1] * unit[1:], axis=1)
        return [float(1.0 - c) for c in cos]

    def _group_by_breakpoints(
        self, sentences: list[str], distances: list[float], threshold: float
    ) -> list[str]:
        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        for i, dist in enumerate(distances):
            if dist > threshold:
                chunks.append(" ".join(current))
                current = []
            current.append(sentences[i + 1])
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _enforce_max_size(self, chunks: list[str]) -> list[str]:
        out: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self._max_chunk_chars:
                out.append(chunk)
            else:
                out.extend(self._fallback.split(chunk))
        return out
