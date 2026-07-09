"""
app/gateway/routes/rag.py

RAG API: ingest documents, query the corpus, list ingested documents.

Routes are thin — all logic lives in the RAG services (ingestion pipeline,
hybrid retriever, query transformer), injected from app.state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Query

from app.core.di import get_state
from app.infra.db.unit_of_work import UnitOfWork
from app.schemas.rag import (
    DocumentDTO,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

if TYPE_CHECKING:
    from app.infra.clients.postgres import PostgresClient
    from app.services.rag.ingestion import IngestionPipeline
    from app.services.rag.query_transform import QueryTransformer
    from app.services.rag.retrieval import HybridRetriever, RetrievedChunk

router = APIRouter(prefix="/rag", tags=["rag"])

IngestionDep = Annotated["IngestionPipeline", Depends(get_state("rag_ingestion"))]
RetrieverDep = Annotated["HybridRetriever", Depends(get_state("rag_retriever"))]
TransformerDep = Annotated["QueryTransformer", Depends(get_state("rag_query_transformer"))]
PostgresDep = Annotated["PostgresClient", Depends(get_state("postgres"))]


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, pipeline: IngestionDep) -> IngestResponse:
    """Ingest a source (file/URL) or raw text into the corpus."""
    if req.source is not None:
        result = await pipeline.ingest_source(req.source, force=req.force)
    else:
        assert req.text is not None  # guaranteed by IngestRequest validator
        result = await pipeline.ingest_text(
            req.text,
            source_uri=req.title or "inline-text",
            title=req.title,
            doc_type=req.doc_type,
            metadata=req.metadata,
            force=req.force,
        )
    return IngestResponse(**result.model_dump())


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    retriever: RetrieverDep,
    transformer: TransformerDep,
) -> QueryResponse:
    """Retrieve the most relevant chunks for a query."""
    if req.use_multi_query:
        results = await _multi_query_retrieve(req, retriever, transformer)
    else:
        search_text = await transformer.hyde(req.query) if req.use_hyde else req.query
        results = await retriever.retrieve(
            search_text,
            top_k=req.top_k,
            filters=req.filters,
            mode=req.mode,
            use_rerank=req.use_rerank,
        )
    return QueryResponse(query=req.query, count=len(results), results=results)


async def _multi_query_retrieve(
    req: QueryRequest,
    retriever: HybridRetriever,
    transformer: QueryTransformer,
) -> list[RetrievedChunk]:
    subqueries = await transformer.multi_query(req.query)
    merged: dict[str, RetrievedChunk] = {}
    for subquery in subqueries:
        search_text = await transformer.hyde(subquery) if req.use_hyde else subquery
        for chunk in await retriever.retrieve(
            search_text,
            top_k=req.top_k,
            filters=req.filters,
            mode=req.mode,
            use_rerank=req.use_rerank,
        ):
            if chunk.chunk_id not in merged or chunk.score > merged[chunk.chunk_id].score:
                merged[chunk.chunk_id] = chunk
    ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)
    return ranked[: req.top_k]


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    postgres: PostgresDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    """List ingested documents, newest first."""
    async with UnitOfWork(postgres) as uow:
        total = await uow.documents.count()
        documents = await uow.documents.list_documents(limit=limit, offset=offset)
        items = [
            DocumentDTO(
                id=str(doc.id),
                source_uri=doc.source_uri,
                title=doc.title,
                doc_type=doc.doc_type,
                chunk_count=len(doc.chunks),
                created_at=doc.created_at,
                metadata=doc.meta,
            )
            for doc in documents
        ]
    return DocumentListResponse(items=items, total=total, limit=limit, offset=offset)
