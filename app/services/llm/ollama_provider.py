"""
app/services/llm/ollama_provider.py

LLMProvider implementation backed by OllamaClient.
Handles native function calling, token streaming and embedding.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.core.config import LLMSettings
from app.core.logging import get_logger
from app.infra.clients.ollama import OllamaClient
from app.services.llm.base import LLMMessage, LLMResponse, StreamChunk

logger = get_logger(__name__)


class OllamaProvider:
    """
    Wraps OllamaClient into the LLMProvider interface.

    Tool calling:
        Ollama returns tool_calls as a list inside message.tool_calls.
        We normalise them to the common dict format used across providers.

    Streaming:
        Yields StreamChunk per token. The final chunk (done=True) carries
        token counts. Streaming does not support tool calls — use chat().
    """

    def __init__(self, client: OllamaClient, settings: LLMSettings) -> None:
        self._client = client
        self._model = settings.main_model
        self._embed_model = settings.embed_model

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raw = await self._client.chat(
            model=self._model,
            messages=messages,
            tools=tools,
            stream=False,
            options=options,
        )
        msg = raw.get("message", {})
        content: str = msg.get("content", "") or ""

        # Normalise Ollama tool_calls → [{name, arguments}]
        raw_tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []
        tool_calls = [
            {
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", {}),
            }
            for tc in raw_tool_calls
        ]

        usage = raw.get("usage") or {}  # not always present
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=raw.get("model", self._model),
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        async for chunk in self._client.stream_chat(
            model=self._model,
            messages=messages,
            options=options,
        ):
            msg = chunk.get("message", {})
            token: str = msg.get("content", "") or ""
            done: bool = chunk.get("done", False)

            if done:
                # Final chunk — extract token counts
                yield StreamChunk(
                    token=token,
                    done=True,
                    prompt_tokens=chunk.get("prompt_eval_count"),
                    completion_tokens=chunk.get("eval_count"),
                )
            else:
                yield StreamChunk(token=token, done=False)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._client.embed(model=self._embed_model, input=texts)