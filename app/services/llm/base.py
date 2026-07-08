"""
app/services/llm/base.py

LLMProvider Protocol — the single contract all LLM backends implement.

Why Protocol (not ABC)?
  Protocol enables structural subtyping: any object that satisfies the
  interface works, without inheriting from a base class. This makes
  mocking trivial (no import of the base needed) and lets cloud SDKs
  be wrapped without monkey-patching.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# Type alias for a single LLM message
LLMMessage = dict[str, Any]


class StreamChunk(BaseModel):
    """One token event from a streaming response."""

    token: str
    done: bool = False
    # Present only on the final chunk (done=True)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMResponse(BaseModel):
    """Complete (non-streaming) response from the LLM."""

    content: str
    tool_calls: list[dict[str, Any]] = []
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    """
    Contract for any LLM backend (Ollama / Anthropic / OpenAI / vLLM).

    Implementations live in:
        app/services/llm/ollama_provider.py
        app/services/llm/cloud_provider.py
    """

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Single-turn, non-streaming completion."""
        ...

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Token-by-token streaming completion."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of texts."""
        ...