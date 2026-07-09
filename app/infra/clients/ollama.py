"""
app/infra/clients/ollama.py

Low-level async HTTP client for the Ollama REST API.
Handles retries, timeouts, and raw streaming — no business logic here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.exceptions import LLMError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.core.config import OllamaSettings

logger = get_logger(__name__)


class OllamaClient:
    """
    Thin async wrapper around the Ollama HTTP API.

    One instance is shared across the process (singleton via lifespan).
    All methods raise :class:`~app.core.exceptions.LLMError` on unrecoverable failures.
    """

    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(settings.timeout_s),
                write=30.0,
                pool=10.0,
            ),
            headers={"Content-Type": "application/json"},
        )


    # Lifecycle

    async def aclose(self) -> None:
        await self._client.aclose()


    # Chat completions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Non-streaming chat completion.

        Returns the full parsed JSON response from Ollama.
        For streaming use :meth:`stream_chat`.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Ollama chat HTTP error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Ollama chat request failed: {exc}") from exc

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Streaming chat — yields one parsed JSON chunk per token event.

        The last chunk has ``"done": true`` and contains final stats.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    if raw_line.strip():
                        yield json.loads(raw_line)
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Ollama stream HTTP error {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Ollama stream request failed: {exc}") from exc


    # Embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def embed(self, model: str, input: str | list[str]) -> list[list[float]]:
        """
        Generate dense embeddings.

        Returns a list of float vectors, one per input string.
        """
        payload: dict[str, Any] = {"model": model, "input": input}
        try:
            response = await self._client.post("/api/embed", json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data["embeddings"]  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Ollama embed HTTP error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Ollama embed request failed: {exc}") from exc


    # Health

    async def ping(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            response = await self._client.get("/")
            return response.status_code == 200
        except httpx.RequestError:
            return False
