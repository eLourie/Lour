"""
app/schemas/rag.py

API DTOs for the RAG routes (/v1/rag/{ingest,query,documents}).
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — runtime type needed by Pydantic
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.services.rag.retrieval import RetrievalMode, RetrievedChunk


class IngestRequest(BaseModel):
    """Ingest either a *source* (file path / URL) or raw *text* — exactly one."""

    source: str | None = Field(default=None, description="File path or URL to load.")
    text: str | None = Field(default=None, description="Raw text to ingest directly.")
    title: str | None = Field(default=None, description="Title (raw-text mode only).")
    doc_type: str = Field(default="text", description="Document type (raw-text mode).")
    metadata: dict[str, Any] = Field(default_factory=dict)
    force: bool = Field(default=False, description="Re-ingest even if content is unchanged.")

    @model_validator(mode="after")
    def _exactly_one_input(self) -> IngestRequest:
        if bool(self.source) == bool(self.text):
            raise ValueError("provide exactly one of 'source' or 'text'")
        return self


class IngestResponse(BaseModel):
    document_id: str | None
    source_uri: str
    chunks: int
    skipped: bool
    reason: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Payload equality filters, e.g. {'doc_type': 'markdown'}.",
    )
    mode: RetrievalMode = RetrievalMode.HYBRID
    use_rerank: bool = True
    use_hyde: bool = Field(default=False, description="Query via a hypothetical answer (HyDE).")
    use_multi_query: bool = Field(default=False, description="Expand into multiple rewrites.")


class QueryResponse(BaseModel):
    query: str
    count: int
    results: list[RetrievedChunk]


class DocumentDTO(BaseModel):
    id: str
    source_uri: str
    title: str | None
    doc_type: str
    chunk_count: int
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    items: list[DocumentDTO]
    total: int
    limit: int
    offset: int
