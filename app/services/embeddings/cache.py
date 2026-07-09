"""
app/services/embeddings/cache.py

Redis-backed embedding cache keyed by SHA-256 of the input text.

Embeddings are expensive (Ollama round-trip). Caching prevents
re-computing identical texts across requests and ingestion runs.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.infra.clients.redis import RedisClient
    from app.services.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days — embeddings rarely change


def _cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return f"emb:{digest}"


class CachedEmbeddingService:
    """
    Wraps any EmbeddingProvider with a Redis cache.

    Cache key = ``emb:<sha256(text)>``.
    On cache miss: embed via the inner provider, store result, return.
    On cache hit: deserialise JSON and return without touching the LLM.
    """

    def __init__(self, inner: EmbeddingProvider, redis: RedisClient) -> None:
        self._inner = inner
        self._redis = redis

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            raw = await self._redis.cache.get(_cache_key(text))
            if raw is not None:
                results[i] = json.loads(raw)
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        # Batch-embed cache misses
        if miss_texts:
            vectors = await self._inner.embed(miss_texts)
            for idx, vec in zip(miss_indices, vectors, strict=True):
                results[idx] = vec
                await self._redis.cache.set(
                    _cache_key(texts[idx]),
                    json.dumps(vec),
                    ex=_TTL_SECONDS,
                )
            logger.debug("embedding_cache_miss", count=len(miss_texts))

        # At this point all results are filled — cast away None
        return [v for v in results if v is not None]

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions
