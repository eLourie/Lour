"""
tests/unit/test_chunking.py

Unit tests for the chunkers — no I/O, no embeddings.
"""

from __future__ import annotations

import pytest

from app.services.rag.chunking import RecursiveCharacterChunker, SemanticChunker

pytestmark = pytest.mark.unit


def test_recursive_empty_returns_empty() -> None:
    assert RecursiveCharacterChunker().split("") == []
    assert RecursiveCharacterChunker().split("   \n  ") == []


def test_recursive_short_text_single_chunk() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.split("A short paragraph.")
    assert chunks == ["A short paragraph."]


def test_recursive_respects_max_size() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=20)
    text = "Sentence number one is here. " * 40
    chunks = chunker.split(text)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_recursive_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)


async def test_semantic_without_embedder_falls_back() -> None:
    # No embedder → deterministic recursive fallback.
    chunker = SemanticChunker(embedder=None, max_chunk_chars=120)
    text = "One two three four. " * 30
    chunks = await chunker.split(text)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


async def test_semantic_embed_failure_falls_back() -> None:
    class BrokenEmbedder:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding backend down")

        @property
        def dimensions(self) -> int:
            return 3

    chunker = SemanticChunker(embedder=BrokenEmbedder(), max_chunk_chars=200, min_sentences=1)
    chunks = await chunker.split("First sentence. Second sentence. Third sentence.")
    assert chunks  # fell back, did not raise


async def test_semantic_breaks_on_topic_shift() -> None:
    # Fake embedder: two tight clusters that are orthogonal → one breakpoint.
    class ClusterEmbedder:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            out: list[list[float]] = []
            for t in texts:
                out.append([1.0, 0.0] if "cat" in t else [0.0, 1.0])
            return out

        @property
        def dimensions(self) -> int:
            return 2

    chunker = SemanticChunker(
        embedder=ClusterEmbedder(),
        max_chunk_chars=10_000,
        min_sentences=2,
        breakpoint_percentile=50.0,
    )
    text = "The cat sat. The cat ran. The dog barked. The dog slept."
    chunks = await chunker.split(text)
    assert len(chunks) >= 2
