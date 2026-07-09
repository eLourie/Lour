"""
app/services/rag/retrieval.py

Hybrid retriever: dense + BM42 sparse → RRF fusion (server-side in Qdrant) →
optional cross-encoder rerank → top-K, with metadata filtering.

Design notes:
  - Fusion (Reciprocal Rank Fusion) happens inside Qdrant via a single
    ``query_points`` call with two prefetch legs. This avoids shipping two
    candidate lists back to the app just to merge them.
  - Reranking is a *reordering* step on the fused candidates. The reranker
    degrades gracefully (returns equal scores when the MPS service is down),
    in which case the RRF order is preserved.
  - ``mode="dense"`` bypasses sparse + fusion — used by the eval suite to show
    hybrid beats pure dense.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
)
from qdrant_client.http.models import SparseVector as QSparseVector

from app.core.logging import get_logger
from app.core.telemetry import traced
from app.infra.clients.qdrant import DENSE_VECTOR, SPARSE_VECTOR

if TYPE_CHECKING:
    from app.infra.clients.qdrant import QdrantClient
    from app.services.embeddings.base import EmbeddingProvider
    from app.services.embeddings.sparse import SparseEmbeddingProvider
    from app.services.reranker.base import Reranker

logger = get_logger(__name__)


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    DENSE = "dense"


class RetrievedChunk(BaseModel):
    """A single retrieval result."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    source_uri: str | None = None
    title: str | None = None
    doc_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HybridRetriever:
    """Dense + sparse hybrid retrieval with RRF fusion and optional rerank."""

    def __init__(
        self,
        *,
        qdrant: QdrantClient,
        dense_embedder: EmbeddingProvider,
        sparse_embedder: SparseEmbeddingProvider,
        reranker: Reranker | None = None,
        collection: str,
        candidate_multiplier: int = 4,
    ) -> None:
        self._qdrant = qdrant
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._reranker = reranker
        self._collection = collection
        self._candidate_multiplier = candidate_multiplier

    @traced("rag_retrieval")
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        use_rerank: bool = True,
    ) -> list[RetrievedChunk]:
        query_filter = self._build_filter(filters)
        # Over-fetch so the reranker has room to reorder.
        candidate_limit = max(top_k * self._candidate_multiplier, top_k)

        dense_vec = (await self._dense.embed([query]))[0]

        if mode is RetrievalMode.DENSE:
            response = await self._qdrant.client.query_points(
                collection_name=self._collection,
                query=dense_vec,
                using=DENSE_VECTOR,
                limit=candidate_limit,
                query_filter=query_filter,
                with_payload=True,
            )
        else:
            sparse = await self._sparse.embed_query(query)
            response = await self._qdrant.client.query_points(
                collection_name=self._collection,
                prefetch=[
                    Prefetch(query=dense_vec, using=DENSE_VECTOR, limit=candidate_limit),
                    Prefetch(
                        query=QSparseVector(indices=sparse.indices, values=sparse.values),
                        using=SPARSE_VECTOR,
                        limit=candidate_limit,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=candidate_limit,
                query_filter=query_filter,
                with_payload=True,
            )

        candidates = [self._to_chunk(p) for p in response.points]
        if not candidates:
            return []

        if use_rerank and self._reranker is not None:
            candidates = await self._rerank(query, candidates)

        return candidates[:top_k]

    async def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        scores = await self._reranker.rerank(query, [c.content for c in candidates])  # type: ignore[union-attr]
        # All-zero scores → service unavailable; keep the fused order.
        if not any(scores):
            return candidates
        for chunk, score in zip(candidates, scores, strict=True):
            chunk.score = float(score)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
        return Filter(must=conditions)

    @staticmethod
    def _to_chunk(point: Any) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            chunk_id=str(point.id),
            document_id=str(payload.get("document_id", "")),
            content=payload.get("content", ""),
            score=float(point.score) if point.score is not None else 0.0,
            source_uri=payload.get("source_uri"),
            title=payload.get("title"),
            doc_type=payload.get("doc_type"),
            metadata=payload.get("metadata", {}),
        )
