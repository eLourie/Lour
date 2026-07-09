"""
app/services/memory/episodic.py

Episodic memory: an append-only chronological ledger in PostgreSQL.

Where long-term memory answers "what do I know about X?" (unordered, semantic),
episodic memory answers "what happened, and when?" — the timeline of a session
or of the whole instance. It is written through the Repository + UoW stack
(ADR-009) so every append shares the project's transactional discipline.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — runtime-needed by Pydantic EpisodicEntry
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.telemetry import traced
from app.infra.db.models.episodic import EpisodicEvent
from app.infra.db.unit_of_work import UnitOfWork

if TYPE_CHECKING:
    from app.infra.clients.postgres import PostgresClient

logger = get_logger(__name__)


class EpisodicEntry(BaseModel):
    """A read-model view of one timeline event (detached from the ORM)."""

    event_type: str
    role: str | None
    content: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodicMemory:
    """Chronological event ledger over PostgreSQL."""

    def __init__(self, postgres: PostgresClient) -> None:
        self._postgres = postgres

    @traced("memory_episodic_record")
    async def record(
        self,
        session_id: str,
        event_type: str,
        content: str,
        *,
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one event to the ledger."""
        async with UnitOfWork(self._postgres) as uow:
            await uow.episodic.add_event(
                EpisodicEvent(
                    session_id=session_id,
                    event_type=event_type,
                    role=role,
                    content=content,
                    meta=metadata or {},
                )
            )

    async def timeline(
        self,
        *,
        session_id: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[EpisodicEntry]:
        """Return events newest-first, optionally scoped by session and time."""
        async with UnitOfWork(self._postgres) as uow:
            rows = await uow.episodic.timeline(
                session_id=session_id, since=since, limit=limit
            )
            return [
                EpisodicEntry(
                    event_type=row.event_type,
                    role=row.role,
                    content=row.content,
                    created_at=row.created_at,
                    metadata=row.meta,
                )
                for row in rows
            ]
