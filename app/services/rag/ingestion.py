"""
app/services/rag/ingestion.py

Idempotent ingestion pipeline.

Flow:
  1. Resolve a loader for the source and load it → LoadedDocument.
  2. Deduplicate by ``doc_hash`` (SHA-256 of content) — identical content is a
     no-op unless ``force=True``.
  3. Chunk (loader-provided ``segments`` for code, else the semantic chunker).
  4. Embed each chunk twice: dense (bge-m3) and sparse (BM42).
  5. Persist atomically — PostgreSQL rows (Document + Chunks) and Qdrant points
     are written inside one Unit of Work; a failure rolls back the PG side.

Re-ingesting a *changed* source (same URI, new hash) replaces the previous
document: its old chunks and Qdrant points are deleted first.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from qdrant_client.http.models import PointIdsList, PointStruct
from qdrant_client.http.models import SparseVector as QSparseVector

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.core.telemetry import traced
from app.infra.clients.qdrant import DENSE_VECTOR, SPARSE_VECTOR
from app.infra.db.models.document import Chunk, Document
from app.infra.db.unit_of_work import UnitOfWork
from app.services.rag.loaders.base import LoadedDocument

if TYPE_CHECKING:
    from app.infra.clients.postgres import PostgresClient
    from app.infra.clients.qdrant import QdrantClient
    from app.services.embeddings.base import EmbeddingProvider
    from app.services.embeddings.sparse import SparseEmbeddingProvider
    from app.services.rag.chunking import SemanticChunker
    from app.services.rag.loaders.base import Loader

logger = get_logger(__name__)


class IngestResult(BaseModel):
    """Outcome of ingesting one source."""

    document_id: str | None
    source_uri: str
    chunks: int
    skipped: bool = False
    reason: str | None = None


class IngestionPipeline:
    """Loads, chunks, embeds and stores documents into PG + Qdrant."""

    def __init__(
        self,
        *,
        loaders: list[Loader],
        chunker: SemanticChunker,
        dense_embedder: EmbeddingProvider,
        sparse_embedder: SparseEmbeddingProvider,
        qdrant: QdrantClient,
        postgres: PostgresClient,
        collection: str,
    ) -> None:
        self._loaders = loaders
        self._chunker = chunker
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._qdrant = qdrant
        self._postgres = postgres
        self._collection = collection

    def _resolve_loader(self, source: str) -> Loader:
        for loader in self._loaders:
            if loader.supports(source):
                return loader
        raise ValidationError(f"No loader supports source: {source}")

    @traced("rag_ingest_source")
    async def ingest_source(self, source: str, *, force: bool = False) -> IngestResult:
        loader = self._resolve_loader(source)
        loaded = await loader.load(source)
        return await self._ingest(loaded, force=force)

    @traced("rag_ingest_text")
    async def ingest_text(
        self,
        text: str,
        *,
        source_uri: str,
        title: str | None = None,
        doc_type: str = "text",
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> IngestResult:
        loaded = LoadedDocument(
            content=text,
            source_uri=source_uri,
            doc_type=doc_type,
            title=title,
            metadata=metadata or {},
        )
        return await self._ingest(loaded, force=force)

    async def _ingest(self, loaded: LoadedDocument, *, force: bool) -> IngestResult:
        doc_hash = hashlib.sha256(loaded.content.encode("utf-8")).hexdigest()

        # Fast-path dedup: identical content already ingested.
        if not force:
            async with UnitOfWork(self._postgres) as uow:
                existing = await uow.documents.get_by_hash(doc_hash)
            if existing is not None:
                logger.info("ingest_skip_duplicate", source=loaded.source_uri, hash=doc_hash[:12])
                return IngestResult(
                    document_id=str(existing.id),
                    source_uri=loaded.source_uri,
                    chunks=0,
                    skipped=True,
                    reason="duplicate",
                )

        chunk_texts = (
            loaded.segments
            if loaded.segments is not None
            else await self._chunker.split(loaded.content)
        )
        chunk_texts = [c for c in chunk_texts if c.strip()]
        if not chunk_texts:
            return IngestResult(
                document_id=None,
                source_uri=loaded.source_uri,
                chunks=0,
                skipped=True,
                reason="no_content",
            )

        dense_vectors = await self._dense.embed(chunk_texts)
        sparse_vectors = await self._sparse.embed_documents(chunk_texts)

        async with UnitOfWork(self._postgres) as uow:
            old_point_ids = await self._delete_existing(uow, doc_hash, loaded.source_uri)

            document = Document(
                doc_hash=doc_hash,
                source_uri=loaded.source_uri,
                title=loaded.title,
                doc_type=loaded.doc_type,
                meta=loaded.metadata,
            )
            await uow.documents.add(document)

            points: list[PointStruct] = []
            for i, text in enumerate(chunk_texts):
                point_id = uuid.uuid4()
                await uow.documents.add_chunk(
                    Chunk(
                        document_id=document.id,
                        chunk_index=i,
                        content=text,
                        point_id=point_id,
                        meta={},
                    )
                )
                points.append(
                    self._build_point(
                        point_id=point_id,
                        document=document,
                        chunk_index=i,
                        content=text,
                        dense=dense_vectors[i],
                        sparse=sparse_vectors[i],
                        loaded=loaded,
                    )
                )

            document_id = str(document.id)

            # Qdrant writes happen inside the UoW so a failure rolls back PG too.
            if old_point_ids:
                await self._qdrant.client.delete(
                    collection_name=self._collection,
                    points_selector=PointIdsList(points=[str(p) for p in old_point_ids]),
                )
            await self._qdrant.client.upsert(collection_name=self._collection, points=points)

        logger.info(
            "ingest_complete",
            source=loaded.source_uri,
            doc_type=loaded.doc_type,
            chunks=len(chunk_texts),
        )
        return IngestResult(
            document_id=document_id,
            source_uri=loaded.source_uri,
            chunks=len(chunk_texts),
        )

    async def _delete_existing(
        self, uow: UnitOfWork, doc_hash: str, source_uri: str
    ) -> list[uuid.UUID]:
        """Delete any prior document with this hash or source; return old point ids."""
        to_delete: list[Document] = []
        seen: set[uuid.UUID] = set()

        by_hash = await uow.documents.get_by_hash(doc_hash)
        if by_hash is not None:
            to_delete.append(by_hash)
            seen.add(by_hash.id)

        by_source = await uow.documents.get_by_source(source_uri)
        if by_source is not None and by_source.id not in seen:
            to_delete.append(by_source)

        old_point_ids: list[uuid.UUID] = []
        for doc in to_delete:
            old_point_ids.extend(await uow.documents.list_point_ids(doc.id))
            await uow.documents.delete(doc)
        return old_point_ids

    @staticmethod
    def _build_point(
        *,
        point_id: uuid.UUID,
        document: Document,
        chunk_index: int,
        content: str,
        dense: list[float],
        sparse: Any,
        loaded: LoadedDocument,
    ) -> PointStruct:
        return PointStruct(
            id=str(point_id),
            vector={
                DENSE_VECTOR: dense,
                SPARSE_VECTOR: QSparseVector(indices=sparse.indices, values=sparse.values),
            },
            payload={
                "document_id": str(document.id),
                "chunk_index": chunk_index,
                "content": content,
                "source_uri": loaded.source_uri,
                "title": loaded.title,
                "doc_type": loaded.doc_type,
                "metadata": loaded.metadata,
            },
        )
