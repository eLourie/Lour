"""
app/infra/db/repositories/document_repo.py

Typed repository for Document / Chunk models, built on Repository[T] + UoW.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, select

from app.infra.db.models.document import Chunk, Document
from app.infra.db.repositories.base import Repository

if TYPE_CHECKING:
    from uuid import UUID


class DocumentRepository(Repository[Document]):
    model = Document

    async def get_by_hash(self, doc_hash: str) -> Document | None:
        """Return the document with this content hash, if already ingested."""
        stmt = select(Document).where(Document.doc_hash == doc_hash)
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def get_by_source(self, source_uri: str) -> Document | None:
        """Return the most recently ingested document for a source URI."""
        stmt = (
            select(Document)
            .where(Document.source_uri == source_uri)
            .order_by(Document.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def list_documents(self, *, limit: int = 50, offset: int = 0) -> list[Document]:
        """Return a page of documents, newest first."""
        stmt = (
            select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count(self) -> int:
        """Total number of documents in the corpus."""
        stmt = select(func.count()).select_from(Document)
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def list_point_ids(self, document_id: UUID) -> list[UUID]:
        """Return the Qdrant point ids of every chunk of a document."""
        stmt = select(Chunk.point_id).where(Chunk.document_id == document_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def add_chunk(self, chunk: Chunk) -> Chunk:
        """Stage a chunk row and flush."""
        self._session.add(chunk)
        await self._session.flush()
        return chunk
