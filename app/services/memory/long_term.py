"""
app/services/memory/long_term.py

Long-term memory: cross-session semantic recall in Qdrant.

Each memory is a short natural-language fact, stored as a dense-embedded point
in the ``memories`` collection (dense named vector; the sparse leg the collection
also declares is unused here — semantic recall of facts does not benefit from
BM42 the way document retrieval does).

Ranking is *not* pure cosine. A memory's usefulness combines three signals
(§7 Phase 4):

    score = alpha * cosine + beta * recency + gamma * importance

  - cosine    — semantic relevance to the query (Qdrant, clamped to [0, 1])
  - recency   — exponential decay by a configurable half-life (fresh > stale)
  - importance— LLM-judged salience assigned at write time (see scoring.py)

Qdrant does the vector search; the re-scoring is done in-process over an
over-fetched candidate set so recency/importance can reorder the cosine order.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from qdrant_client.http.models import PointStruct

from app.core.logging import get_logger
from app.core.telemetry import traced
from app.infra.clients.qdrant import DENSE_VECTOR

if TYPE_CHECKING:
    from app.core.config import MemorySettings
    from app.infra.clients.qdrant import QdrantClient
    from app.services.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


class MemoryItem(BaseModel):
    """A recalled long-term memory with its scoring breakdown."""

    memory_id: str
    content: str
    score: float  # combined alpha/beta/gamma score
    cosine: float
    importance: float
    created_at: datetime
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongTermMemory:
    """Semantic long-term store with importance + recency re-ranking."""

    def __init__(
        self,
        *,
        qdrant: QdrantClient,
        embedder: EmbeddingProvider,
        settings: MemorySettings,
        collection: str,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._collection = collection
        self._top_k = settings.long_term_top_k
        self._candidate_multiplier = settings.candidate_multiplier
        self._alpha = settings.score_alpha
        self._beta = settings.score_beta
        self._gamma = settings.score_gamma
        self._half_life_h = settings.recency_half_life_h

    @traced("memory_write")
    async def write(
        self,
        content: str,
        *,
        importance: float,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Embed and persist one fact; return its memory id."""
        vector = (await self._embedder.embed([content]))[0]
        memory_id = str(uuid.uuid4())
        now = datetime.now(tz=UTC)
        point = PointStruct(
            id=memory_id,
            vector={DENSE_VECTOR: vector},
            payload={
                "content": content,
                "importance": float(importance),
                "created_at": now.timestamp(),
                "session_id": session_id,
                "metadata": metadata or {},
            },
        )
        await self._qdrant.client.upsert(collection_name=self._collection, points=[point])
        logger.debug("memory_written", memory_id=memory_id, importance=round(importance, 3))
        return memory_id

    @traced("memory_search")
    async def search(self, query: str, *, top_k: int | None = None) -> list[MemoryItem]:
        """Semantic recall, re-ranked by cosine + recency + importance."""
        limit = top_k or self._top_k
        candidate_limit = max(limit * self._candidate_multiplier, limit)
        vector = (await self._embedder.embed([query]))[0]

        response = await self._qdrant.client.query_points(
            collection_name=self._collection,
            query=vector,
            using=DENSE_VECTOR,
            limit=candidate_limit,
            with_payload=True,
        )
        now = datetime.now(tz=UTC)
        items = [self._to_item(point, now) for point in response.points]
        items.sort(key=lambda i: i.score, reverse=True)
        return items[:limit]

    async def nearest_cosine(self, content: str) -> float:
        """Return the cosine of the single closest existing memory (0.0 if empty)."""
        vector = (await self._embedder.embed([content]))[0]
        response = await self._qdrant.client.query_points(
            collection_name=self._collection,
            query=vector,
            using=DENSE_VECTOR,
            limit=1,
            with_payload=False,
        )
        if not response.points:
            return 0.0
        top = response.points[0].score
        return float(top) if top is not None else 0.0

    def _to_item(self, point: Any, now: datetime) -> MemoryItem:
        payload = point.payload or {}
        cosine = max(0.0, float(point.score) if point.score is not None else 0.0)
        importance = float(payload.get("importance", 0.0))
        created_ts = float(payload.get("created_at", now.timestamp()))
        created_at = datetime.fromtimestamp(created_ts, tz=UTC)
        recency = self._recency(created_ts, now.timestamp())
        combined = self._alpha * cosine + self._beta * recency + self._gamma * importance
        return MemoryItem(
            memory_id=str(point.id),
            content=payload.get("content", ""),
            score=combined,
            cosine=cosine,
            importance=importance,
            created_at=created_at,
            session_id=payload.get("session_id"),
            metadata=payload.get("metadata", {}),
        )

    def _recency(self, created_ts: float, now_ts: float) -> float:
        """Exponential decay in [0, 1]: 1.0 when fresh, 0.5 after one half-life."""
        age_hours = max(0.0, (now_ts - created_ts) / 3600.0)
        return float(0.5 ** (age_hours / self._half_life_h))
