"""
app/services/memory/short_term.py

Short-term memory: the current session's working context in Redis.

Model:
  - A per-session sliding window of the most recent turns, kept verbatim as a
    Redis list at ``stm:msgs:{session_id}``.
  - When the window overflows, the oldest turns are evicted and folded into a
    rolling natural-language summary at ``stm:sum:{session_id}`` via the LLM —
    so nothing is silently lost, but the token cost stays bounded.

Everything is TTL'd: an idle session's working set expires on its own, which is
the right default for a single-user instance (no cross-session leakage, §1.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.core.config import MemorySettings
    from app.infra.clients.redis import RedisClient
    from app.services.llm.base import LLMProvider

logger = get_logger(__name__)

_MSGS_PREFIX = "stm:msgs:"
_SUMMARY_PREFIX = "stm:sum:"


class ConversationTurn(BaseModel):
    """One verbatim message in the working window."""

    role: str
    content: str


class ShortTermMemory:
    """Redis sliding window with LLM tail-summarisation."""

    def __init__(
        self,
        redis: RedisClient,
        llm: LLMProvider,
        settings: MemorySettings,
    ) -> None:
        self._redis = redis.memory
        self._llm = llm
        self._window = settings.short_term_window
        self._ttl = settings.short_term_ttl_s

    @staticmethod
    def _msgs_key(session_id: str) -> str:
        return f"{_MSGS_PREFIX}{session_id}"

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f"{_SUMMARY_PREFIX}{session_id}"

    async def append(self, session_id: str, role: str, content: str) -> None:
        """Append a turn; summarise and evict the tail when the window overflows."""
        key = self._msgs_key(session_id)
        turn = ConversationTurn(role=role, content=content)
        await self._redis.rpush(key, turn.model_dump_json().encode())
        await self._redis.expire(key, self._ttl)

        length = await self._redis.llen(key)
        overflow = int(length) - self._window
        if overflow > 0:
            await self._summarise_tail(session_id, overflow)

    async def _summarise_tail(self, session_id: str, overflow: int) -> None:
        """Pop the ``overflow`` oldest turns and fold them into the rolling summary."""
        key = self._msgs_key(session_id)
        raw = await self._redis.lpop(key, overflow)
        if not raw:
            return
        evicted = [ConversationTurn.model_validate_json(item) for item in raw]

        prior = await self.get_summary(session_id)
        transcript = "\n".join(f"{t.role}: {t.content}" for t in evicted)
        prompt = (
            "You maintain a concise running summary of a conversation so older "
            "turns can be dropped without losing context. Update the summary to "
            "incorporate the new (older) turns below. Keep it factual and brief "
            "(a few sentences).\n\n"
            f"Current summary:\n{prior or '(none yet)'}\n\n"
            f"Older turns to fold in:\n{transcript}\n\n"
            "Updated summary:"
        )
        response = await self._llm.chat([{"role": "user", "content": prompt}])
        summary = response.content.strip()
        if summary:
            await self._redis.set(
                self._summary_key(session_id), summary.encode(), ex=self._ttl
            )
            logger.debug("stm_tail_summarised", session_id=session_id, evicted=len(evicted))

    async def get_window(self, session_id: str) -> list[ConversationTurn]:
        """Return the verbatim turns currently in the window (oldest first)."""
        raw = await self._redis.lrange(self._msgs_key(session_id), 0, -1)
        return [ConversationTurn.model_validate_json(item) for item in raw]

    async def get_summary(self, session_id: str) -> str | None:
        """Return the rolling summary of evicted turns, if any."""
        raw = await self._redis.get(self._summary_key(session_id))
        return raw.decode() if raw is not None else None

    async def clear(self, session_id: str) -> None:
        """Drop the window and summary for a session."""
        await self._redis.delete(self._msgs_key(session_id), self._summary_key(session_id))

    async def active_sessions(self) -> AsyncIterator[str]:
        """Yield the session_id of every session with a live working window."""
        async for key in self._redis.scan_iter(match=f"{_MSGS_PREFIX}*"):
            yield key.decode().removeprefix(_MSGS_PREFIX)
