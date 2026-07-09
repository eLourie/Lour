"""
app/services/memory/base.py

MemoryManager — the single facade the rest of the system talks to.

Memory is a *service*, not something baked into the agent (ADR-005): the graph's
memory nodes (Phase 5) and any script call this facade, which fans out to the
three layers:

  - short-term (Redis)  — the current session's working window + rolling summary
  - long-term (Qdrant)  — cross-session semantic facts, importance/recency ranked
  - episodic (Postgres) — the chronological event ledger

``write`` records a turn into short-term + episodic as it happens; ``recall``
assembles the working window, its summary and the top semantic memories for a
query into one ``MemoryContext``. Long-term *distillation* is asynchronous and
owned by consolidation (ADR-012), not by ``write`` on the hot path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.telemetry import traced

# Runtime imports — MemoryContext is a Pydantic model whose fields reference
# these types, so they must be resolvable at runtime (not just for type checking).
from app.services.memory.long_term import MemoryItem  # noqa: TC001 — Pydantic runtime field
from app.services.memory.short_term import (  # noqa: TC001 — Pydantic runtime field
    ConversationTurn,
)

if TYPE_CHECKING:
    from app.services.memory.episodic import EpisodicMemory
    from app.services.memory.long_term import LongTermMemory
    from app.services.memory.short_term import ShortTermMemory

logger = get_logger(__name__)


class MemoryContext(BaseModel):
    """Everything recall assembles for one query, ready to inject into a prompt."""

    summary: str | None = None
    recent: list[ConversationTurn] = Field(default_factory=list)
    long_term: list[MemoryItem] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.summary and not self.recent and not self.long_term


class MemoryManager:
    """Facade over short-term, long-term and episodic memory."""

    def __init__(
        self,
        *,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        episodic: EpisodicMemory,
    ) -> None:
        self._short_term = short_term
        self._long_term = long_term
        self._episodic = episodic

    @traced("memory_write")
    async def write(self, session_id: str, role: str, content: str) -> None:
        """Record a conversational turn into short-term + episodic memory."""
        await self._short_term.append(session_id, role, content)
        await self._episodic.record(session_id, event_type="message", content=content, role=role)

    @traced("memory_recall")
    async def recall(
        self,
        session_id: str,
        query: str,
        *,
        top_k: int | None = None,
    ) -> MemoryContext:
        """Assemble working context + semantic memories for *query*."""
        summary = await self._short_term.get_summary(session_id)
        recent = await self._short_term.get_window(session_id)
        long_term = await self._long_term.search(query, top_k=top_k)

        # memory hit rate — how often recall surfaces cross-session knowledge.
        logger.info(
            "memory_recall",
            session_id=session_id,
            long_term_hits=len(long_term),
            hit=bool(long_term),
            window=len(recent),
            has_summary=summary is not None,
        )
        return MemoryContext(summary=summary, recent=recent, long_term=long_term)

    async def remember_fact(
        self,
        content: str,
        *,
        importance: float,
        session_id: str | None = None,
    ) -> str:
        """Write a fact straight to long-term memory (bypasses consolidation)."""
        return await self._long_term.write(
            content, importance=importance, session_id=session_id
        )

    async def clear_session(self, session_id: str) -> None:
        """Drop a session's short-term working set (long-term is durable)."""
        await self._short_term.clear(session_id)
