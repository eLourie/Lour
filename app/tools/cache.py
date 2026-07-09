"""
app/tools/cache.py

Redis-backed cache for idempotent tool results.

Only tools with ``side_effects=False`` are cacheable — re-running a search or a
datetime lookup with identical args should be cheap, but a write must always
execute. The cache key is derived from the tool name + a stable hash of the
arguments, so identical calls collapse to one backend round-trip.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.tools.base import ToolResult

if TYPE_CHECKING:
    from app.infra.clients.redis import RedisClient

logger = get_logger(__name__)


def cache_key(tool_name: str, args: dict[str, object]) -> str:
    """Stable key: ``tool:<name>:<sha256 of canonical-json args>``."""
    canonical = json.dumps(args, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"tool:{tool_name}:{digest}"


class ToolResultCache:
    """Stores/loads ToolResult payloads in the Redis cache DB."""

    def __init__(self, redis: RedisClient, *, ttl_s: int = 300) -> None:
        self._redis = redis
        self._ttl_s = ttl_s

    async def get(self, tool_name: str, args: dict[str, object]) -> ToolResult | None:
        raw = await self._redis.cache.get(cache_key(tool_name, args))
        if raw is None:
            return None
        try:
            return ToolResult.model_validate_json(raw)
        except ValueError:
            # Corrupt / schema-drifted entry — treat as a miss.
            return None

    async def set(self, tool_name: str, args: dict[str, object], result: ToolResult) -> None:
        # Never cache failures — a transient error must not stick.
        if not result.ok:
            return
        await self._redis.cache.set(
            cache_key(tool_name, args),
            result.model_dump_json(),
            ex=self._ttl_s,
        )
