"""
app/infra/clients/reranker.py

HTTP client for the native MPS rerank service (runs outside Ollama,
outside Docker — on the Mac host directly via Metal).

The service is expected to expose a simple POST /rerank endpoint:
    Request:  {"query": str, "documents": [str, ...]}
    Response: {"scores": [float, ...]}   # same order as documents
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from app.core.exceptions import RerankerError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.config import RerankerSettings

logger = get_logger(__name__)


class RerankerClient:
    """
    Thin async HTTP client for the local MPS cross-encoder service.

    The actual model (bge-reranker-v2-m3) runs in a separate Python
    process on the host — see ``RERANKER_BASE_URL`` in ``.env``.
    This client only speaks HTTP; it has no model loading logic.
    """

    def __init__(self, settings: RerankerSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            headers={"Content-Type": "application/json"},
        )

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        Return relevance scores for each document relative to *query*.

        Scores are in the same order as *documents*. Higher = more relevant.
        Raises :class:`RerankerError` if the service is unreachable or returns
        a non-2xx status.
        """
        if not documents:
            return []

        payload = {"query": query, "documents": documents}
        try:
            response = await self._client.post("/rerank", json=payload)
            response.raise_for_status()
            data: dict[str, list[float]] = response.json()
            return data["scores"]
        except httpx.HTTPStatusError as exc:
            raise RerankerError(
                f"Reranker HTTP error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise RerankerError(f"Reranker request failed: {exc}") from exc

    async def ping(self) -> bool:
        """Return True if the rerank service is reachable."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
