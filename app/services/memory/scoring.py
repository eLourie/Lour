"""
app/services/memory/scoring.py

Importance scoring via LLM-as-judge.

Not every fact deserves to be remembered forever. Before a candidate fact is
written to long-term memory, an LLM judges how salient / durable it is on a
0-1 scale (a stable preference or identity fact scores high; small talk scores
low). That score feeds the ``gamma * importance`` term of long-term ranking and
the ``min_importance`` write threshold used by consolidation.

The judgement is deterministic-ish and pure per text, so results are cached in
Redis keyed by content hash — re-scoring identical text is free.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.infra.clients.redis import RedisClient
    from app.services.llm.structured import StructuredOutputService

logger = get_logger(__name__)

_CACHE_PREFIX = "imp:"
_CACHE_TTL_S = 60 * 60 * 24 * 30  # 30 days

_JUDGE_SYSTEM = (
    "You rate how important a piece of information is to remember long-term "
    "about a user or their work, on a scale from 0.0 to 1.0. Durable facts, "
    "stable preferences, identities, decisions and goals score high (0.7-1.0). "
    "Transient chit-chat, acknowledgements and one-off trivia score low "
    "(0.0-0.3). Respond with the score and a short reason."
)


class ImportanceJudgment(BaseModel):
    """Structured verdict from the LLM judge."""

    importance: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class ImportanceScorer:
    """LLM-as-judge importance scoring with a Redis content-hash cache."""

    def __init__(
        self,
        structured: StructuredOutputService,
        redis: RedisClient,
    ) -> None:
        self._structured = structured
        self._redis = redis.memory

    @staticmethod
    def _cache_key(text: str) -> str:
        return f"{_CACHE_PREFIX}{hashlib.sha256(text.encode()).hexdigest()}"

    async def score(self, text: str) -> float:
        """Return an importance in [0, 1] for *text* (cached by content hash)."""
        key = self._cache_key(text)
        cached = await self._redis.get(key)
        if cached is not None:
            return float(cached.decode())

        judgment = await self._structured.complete(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": f"Rate this information:\n{text}"},
            ],
            schema=ImportanceJudgment,
        )
        importance = max(0.0, min(1.0, judgment.importance))
        await self._redis.set(key, str(importance).encode(), ex=_CACHE_TTL_S)
        logger.debug("importance_scored", importance=round(importance, 3))
        return importance
