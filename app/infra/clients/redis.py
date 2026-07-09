"""
app/infra/clients/redis.py

Async Redis client with logical-DB selection and connection pooling.
Exposes three pre-wired clients for cache / memory / rate-limit DBs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import redis.asyncio as aioredis

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.config import RedisSettings

logger = get_logger(__name__)


class RedisClient:
    """
    Thin wrapper that owns a pool and exposes typed logical-DB handles.

    Usage::

        client = RedisClient(settings)
        await client.cache.set("key", "value", ex=60)
        await client.memory.hset("session:1", mapping={"k": "v"})
        await client.rate_limit.incr("rl:user:1")
    """

    def __init__(self, settings: RedisSettings) -> None:
        self._settings = settings

        # One pool per logical DB. Redis connection pools are lightweight.
        self.cache: aioredis.Redis[bytes] = aioredis.from_url(
            settings.url(settings.db_cache),
            encoding="utf-8",
            decode_responses=False,  # raw bytes — callers decode themselves
            max_connections=20,
        )
        self.memory: aioredis.Redis[bytes] = aioredis.from_url(
            settings.url(settings.db_memory),
            encoding="utf-8",
            decode_responses=False,
            max_connections=20,
        )
        self.rate_limit: aioredis.Redis[bytes] = aioredis.from_url(
            settings.url(settings.db_ratelimit),
            encoding="utf-8",
            decode_responses=False,
            max_connections=10,
        )

    async def ping_all(self) -> bool:
        """Return True only if all three logical DBs respond."""
        try:
            await self.cache.ping()
            await self.memory.ping()
            await self.rate_limit.ping()
            return True
        except Exception:
            logger.exception("Redis ping failed")
            return False

    async def aclose(self) -> None:
        await self.cache.close()
        await self.memory.close()
        await self.rate_limit.close()
