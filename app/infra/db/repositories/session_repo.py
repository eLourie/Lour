"""
app/infra/db/repositories/session_repo.py

Typed repository for Session / Message models.
"""

from __future__ import annotations

from sqlalchemy import select

from app.infra.db.models.session import Message, Session
from app.infra.db.repositories.base import Repository


class SessionRepository(Repository[Session]):
    model = Session

    async def get_by_thread_id(self, thread_id: str) -> Session | None:
        """Fetch a session by its LangGraph thread_id."""
        stmt = select(Session).where(Session.thread_id == thread_id)
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def add_message(self, message: Message) -> Message:
        """Append a message and flush."""
        self._session.add(message)
        await self._session.flush()
        return message
