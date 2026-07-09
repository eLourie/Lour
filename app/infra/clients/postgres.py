"""
app/infra/clients/postgres.py

Async SQLAlchemy 2.0 engine + session factory backed by asyncpg.
Two engines: main app DB and LangGraph checkpoint DB (separate database).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.core.config import PostgresSettings

logger = get_logger(__name__)


class PostgresClient:
    """
    Owns two async engines (app DB + checkpoint DB) and session factories.

    Engines are created once at startup and closed on shutdown via
    :meth:`aclose`. Sessions are obtained through the context-manager
    helpers :meth:`session` and :meth:`checkpoint_session`.
    """

    def __init__(self, settings: PostgresSettings) -> None:
        self._settings = settings

        self.engine: AsyncEngine = create_async_engine(
            settings.dsn,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # detect stale connections
            echo=False,
        )
        self.checkpoint_engine: AsyncEngine = create_async_engine(
            settings.checkpoint_dsn,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )

        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        self._checkpoint_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self.checkpoint_engine,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a transactional session for the main app DB."""
        async with self._session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    @asynccontextmanager
    async def checkpoint_session(self) -> AsyncIterator[AsyncSession]:
        """Yield a transactional session for the LangGraph checkpoint DB."""
        async with self._checkpoint_session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    async def ping(self) -> bool:
        """Return True if the main DB is reachable."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            return True
        except Exception:
            logger.exception("Postgres ping failed")
            return False

    async def aclose(self) -> None:
        await self.engine.dispose()
        await self.checkpoint_engine.dispose()
