"""
app/infra/db/repositories/episodic_repo.py

Typed repository for the EpisodicEvent timeline, built on Repository[T] + UoW.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.infra.db.models.episodic import EpisodicEvent
from app.infra.db.repositories.base import Repository

if TYPE_CHECKING:
    from datetime import datetime


class EpisodicRepository(Repository[EpisodicEvent]):
    model = EpisodicEvent

    async def add_event(self, event: EpisodicEvent) -> EpisodicEvent:
        """Append an event to the ledger and flush."""
        self._session.add(event)
        await self._session.flush()
        return event

    async def timeline(
        self,
        *,
        session_id: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[EpisodicEvent]:
        """
        Return events newest-first, optionally scoped to one session and/or a
        lower time bound.
        """
        stmt = select(EpisodicEvent)
        if session_id is not None:
            stmt = stmt.where(EpisodicEvent.session_id == session_id)
        if since is not None:
            stmt = stmt.where(EpisodicEvent.created_at >= since)
        stmt = stmt.order_by(EpisodicEvent.created_at.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
