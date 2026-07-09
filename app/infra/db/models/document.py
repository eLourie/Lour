"""
app/infra/db/models/document.py

ORM models for the RAG corpus.

Split of responsibilities (twelve-factor / ADR-010):
  - PostgreSQL holds *metadata and text* — the source of truth for what was
    ingested, deduplication hashes, and chunk bookkeeping.
  - Qdrant holds the *vectors* (dense + sparse). Each Chunk row carries the
    ``point_id`` that links it to its Qdrant point, so the two stores stay
    consistent and re-ingestion can delete the right points.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infra.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Document(Base):
    """
    A single ingested source document (pdf / markdown / code / url / text).

    ``doc_hash`` is the SHA-256 of the raw content and is unique — it makes
    ingestion idempotent (re-ingesting identical content is a no-op) and lets
    a changed source replace its previous chunks.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    source_uri: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    doc_type: Mapped[str] = mapped_column(String(32), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Chunk.chunk_index",
    )


class Chunk(Base):
    """
    A retrievable chunk of a Document.

    ``point_id`` is the UUID of the corresponding point in Qdrant (both named
    vectors, dense + sparse, live on that one point).
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_id_chunk_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    point_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    document: Mapped[Document] = relationship("Document", back_populates="chunks")
