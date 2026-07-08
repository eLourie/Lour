"""
app/infra/db/repositories/base.py

Generic async Repository[T] over SQLAlchemy 2.0.

Why Repository pattern here?
- Decouples domain logic from SQLAlchemy — tests can mock the repo,
  not the DB session.
- Makes transactions explicit: the session lives in UoW, not in the repo.
- Generic[T] means one base handles all model types.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class Repository(Generic[ModelT]):
    """
    Base repository with typed CRUD for a single SQLAlchemy model.

    The session is injected on construction (owned by UoW, not here).
    All methods are async; none commit — that is the UoW's job.

    Usage::

        class SessionRepository(Repository[Session]):
            model = Session

        async with uow:
            session = await uow.sessions.get(session_id)
    """

    model: type[ModelT]  # subclasses set this as a class attribute

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, id: UUID) -> ModelT | None:
        """Fetch by primary key; return None if not found."""
        return await self._session.get(self.model, id)

    async def get_by(self, **kwargs: Any) -> ModelT | None:  # noqa: ANN401
        """Fetch first row matching all given column=value filters."""
        stmt = select(self.model).filter_by(**kwargs)
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def list_all(self) -> list[ModelT]:
        """Return all rows (use with care on large tables)."""
        stmt = select(self.model)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def add(self, instance: ModelT) -> ModelT:
        """Stage a new instance; flush so the DB assigns IDs."""
        self._session.add(instance)
        await self._session.flush()
        return instance

    async def delete(self, instance: ModelT) -> None:
        """Mark instance for deletion on next commit."""
        await self._session.delete(instance)
        await self._session.flush()