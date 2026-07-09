"""
app/services/reranker/local_mps.py

Local MPS reranker: delegates to the native MPS service via RerankerClient.

The actual model (bge-reranker-v2-m3) runs in a separate host process.
RERANKER_LAZY_LOAD=true means we don't fail at startup if it's not up yet —
we just mark the reranker as unavailable and skip reranking when called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.infra.clients.reranker import RerankerClient, RerankerError

if TYPE_CHECKING:
    from app.core.config import RerankerSettings

logger = get_logger(__name__)


class LocalMPSReranker:
    """
    Calls the local MPS rerank service.

    With RERANKER_LAZY_LOAD=true the service may not be running at startup.
    :meth:`rerank` degrades gracefully by returning equal scores (0.0)
    when the service is unreachable, rather than crashing the request.
    """

    def __init__(self, client: RerankerClient, settings: RerankerSettings) -> None:
        self._client = client
        self._lazy = settings.lazy_load
        self._ready: bool | None = None  # None = not yet checked

    async def _check_ready(self) -> bool:
        if self._ready is None:
            self._ready = await self._client.ping()
            if self._ready:
                logger.info("reranker_ready")
            else:
                logger.warning("reranker_unavailable", lazy=self._lazy)
        return self._ready

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not await self._check_ready():
            logger.warning("reranker_skip_unavailable", doc_count=len(documents))
            return [0.0] * len(documents)
        try:
            scores = await self._client.rerank(query, documents)
            self._ready = True  # reset on success
            return scores
        except RerankerError as exc:
            logger.error("reranker_error", error=str(exc))
            self._ready = False  # mark for re-check next call
            return [0.0] * len(documents)

    @property
    def available(self) -> bool:
        return self._ready is True
