"""
app/infra/db/unit_of_work.py

Unit of Work — owns the session and exposes typed repositories.

Pattern:
    async with UnitOfWork(postgres_client) as uow:
        session = await uow.sessions.get_by(thread_id="abc")
        ...
        # auto-commit on clean exit, rollback on exception
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from app.infra.db.repositories.document_repo import DocumentRepository
from app.infra.db.repositories.session_repo import SessionRepository

if TYPE_CHECKING:
    from types import TracebackType

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.infra.clients.postgres import PostgresClient


class UnitOfWork:
    """
    Context manager that wraps a single DB transaction.

    Repositories are lazily wired to the session when the context opens.
    Commit / rollback is automatic — callers should never call them directly.
    """

    def __init__(self, postgres: PostgresClient) -> None:
        self._postgres = postgres
        self._session: AsyncSession | None = None
        self.sessions: SessionRepository  # set in __aenter__
        self.documents: DocumentRepository  # set in __aenter__

    async def __aenter__(self) -> Self:
        # We bypass the context-manager on PostgresClient intentionally:
        # UoW controls commit/rollback, not the client's helper.
        self._session = self._postgres._session_factory()
        self.sessions = SessionRepository(self._session)
        self.documents = DocumentRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._session is not None
        if exc_type is None:
            await self._session.commit()
        else:
            await self._session.rollback()
        await self._session.close()
