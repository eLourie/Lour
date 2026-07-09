"""
app/infra/db/models/episodic.py

ORM model for the episodic memory timeline.

Episodic memory is a chronological ledger of what happened — user turns,
agent answers, tool runs, consolidation events. It answers "what did we do
yesterday?" queries that neither the semantic long-term store (unordered) nor
the short-term window (current session only) can serve.

Design:
  - ``session_id`` is a free-form string (the LangGraph thread_id), *not* a FK
    to ``sessions.id``. Episodic events may outlive or precede a Session row and
    the memory layer is deliberately decoupled from orchestration (ADR-005),
    so a hard FK would be the wrong coupling.
  - Rows are append-only; the timeline is queried by ``(session_id, created_at)``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infra.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class EpisodicEvent(Base):
    """
    One entry in the chronological event ledger.

    ``event_type`` is a coarse tag (``message`` / ``tool_call`` / ``fact`` /
    ``consolidation``); ``role`` is set for conversational events
    (user / assistant / tool) and left null otherwise.
    """

    __tablename__ = "episodic_events"
    __table_args__ = (
        # Timeline queries filter by session and order by time.
        Index("ix_episodic_events_session_created", "session_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
