"""
app/services/llm/cloud_provider.py

Reliability-tier LLM provider: Anthropic and OpenAI behind the same
LLMProvider Protocol. Used for long multi-step tool chains where the
local 14B model loses coherence.

The embed() method intentionally raises NotImplementedError — cloud
providers are not used for embeddings (we use local bge-m3 for that).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.core.config import LLMProvider as LLMProviderEnum
from app.core.config import LLMSettings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.services.llm.base import LLMMessage, LLMResponse, StreamChunk

logger = get_logger(__name__)


class AnthropicProvider:
    """
    Anthropic Claude via the official SDK.

    Tool calls follow Anthropic's tool_use block format and are
    normalised to the shared {name, arguments} dict on return.
    """

    def __init__(self, settings: LLMSettings) -> None:
        try:
            import anthropic  # type: ignore[import-untyped]

            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        except ImportError as exc:
            raise LLMError("anthropic package not installed") from exc
        self._model = settings.main_model or "claude-3-5-sonnet-20241022"

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        # Anthropic separates the system prompt from user/assistant turns
        system = ""
        filtered: list[LLMMessage] = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                filtered.append(m)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": (options or {}).get("max_tokens", 4096),
            "messages": filtered,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        import anthropic  # type: ignore[import-untyped]

        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        content = ""
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "arguments": block.input})

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            model=response.model,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError("AnthropicProvider streaming not yet implemented")
        # Makes the method a valid async generator per the Protocol
        yield StreamChunk(token="", done=True)  # pragma: no cover

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Use local bge-m3 for embeddings, not cloud provider")


class OpenAIProvider:
    """OpenAI / OpenAI-compatible (e.g. DeepSeek) provider."""

    def __init__(self, settings: LLMSettings) -> None:
        try:
            import openai  # type: ignore[import-untyped]

            self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        except ImportError as exc:
            raise LLMError("openai package not installed") from exc
        self._model = settings.main_model or "gpt-4o"

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        import openai  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        if options:
            kwargs.update(options)

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls: list[dict[str, Any]] = []
        if msg.tool_calls:
            import json

            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )

        usage = response.usage
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError("OpenAIProvider streaming not yet implemented")
        yield StreamChunk(token="", done=True)  # pragma: no cover

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Use local bge-m3 for embeddings, not cloud provider")


def build_cloud_provider(
    settings: LLMSettings,
) -> AnthropicProvider | OpenAIProvider:
    """Select cloud provider by LLM_PROVIDER env."""
    if settings.provider == LLMProviderEnum.ANTHROPIC:
        return AnthropicProvider(settings)
    if settings.provider == LLMProviderEnum.OPENAI:
        return OpenAIProvider(settings)
    raise LLMError(f"No cloud provider for LLM_PROVIDER={settings.provider!r}")