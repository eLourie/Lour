"""
app/agents/checkpointing.py

Checkpointer wrapper around LangGraph's ``AsyncPostgresSaver`` (ADR-008).

Why Postgres and not in-memory / SQLite: a checkpoint after every node lets a
run be *resumed* after a process restart, *replayed* for debugging and
*time-travelled*. Postgres is already in the stack, and checkpoints live in a
dedicated database (``agent_checkpoint``) so they never collide with app tables.

The saver is backed by a connection pool opened via ``from_conn_string`` (an
async context manager). ``CheckpointerManager`` owns that lifecycle so the app
lifespan can ``start()`` it once and ``aclose()`` it on shutdown, exactly like
the other backing-service clients.

``delete_thread`` gives the retention story a concrete hook: a finished or
pruned session's checkpoints can be dropped explicitly (an admin route or a
future scheduled sweep decides *which* threads, from the sessions ledger).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.config import PostgresSettings

logger = get_logger(__name__)


def _psycopg_dsn(settings: PostgresSettings) -> str:
    """
    Build a libpq/psycopg connection string for the checkpoint database.

    The app's SQLAlchemy DSNs use the ``postgresql+asyncpg`` scheme; psycopg
    (which the LangGraph saver uses) wants a plain ``postgresql://`` URL.
    """
    return (
        f"postgresql://{settings.user}:{settings.password}"
        f"@{settings.host}:{settings.port}/{settings.checkpoint_db}"
    )


class CheckpointerManager:
    """Owns the AsyncPostgresSaver's pool lifecycle for the app lifespan."""

    def __init__(self, settings: PostgresSettings) -> None:
        self._dsn = _psycopg_dsn(settings)
        self._cm: object | None = None
        self._saver: AsyncPostgresSaver | None = None

    async def start(self) -> AsyncPostgresSaver:
        """Open the pool, create the checkpoint tables if needed, return the saver."""
        cm = AsyncPostgresSaver.from_conn_string(self._dsn)
        self._cm = cm
        self._saver = await cm.__aenter__()
        await self._saver.setup()  # idempotent — CREATE TABLE IF NOT EXISTS
        logger.info("checkpointer_ready", db="agent_checkpoint")
        return self._saver

    @property
    def saver(self) -> AsyncPostgresSaver:
        if self._saver is None:
            raise RuntimeError("CheckpointerManager.start() has not been called")
        return self._saver

    async def delete_thread(self, thread_id: str) -> None:
        """Drop all checkpoints for a single thread (retention / privacy)."""
        await self.saver.adelete_thread(thread_id)
        logger.info("checkpoint_thread_deleted", thread_id=thread_id)

    async def aclose(self) -> None:
        """Close the connection pool."""
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)  # type: ignore[attr-defined]
            self._cm = None
            self._saver = None
            logger.info("checkpointer_closed")
