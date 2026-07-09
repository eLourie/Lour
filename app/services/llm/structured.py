"""
app/services/llm/structured.py

Structured output: LLM → validated Pydantic model, via Instructor.

Instructor patches the provider's native SDK client and enforces a
``response_model``. On a ``ValidationError`` it re-prompts the model with
the error (the retry-with-feedback pattern referenced in ADR-007), up to
``MAX_RETRIES`` times, and only returns once the payload validates.

Provider-agnostic by construction:
  - ollama / vllm → reached through the OpenAI-compatible ``/v1`` endpoint
  - openai        → native OpenAI SDK
  - anthropic     → native Anthropic SDK

Changing ``LLM_PROVIDER`` is enough — no code change at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import instructor
from pydantic import BaseModel

from app.core.config import LLMProvider as LLMProviderEnum
from app.core.exceptions import LLMError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.config import LLMSettings, OllamaSettings
    from app.services.llm.base import LLMMessage

logger = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

MAX_RETRIES = 2

# Anthropic's Messages API requires an explicit max_tokens.
_ANTHROPIC_MAX_TOKENS = 4096


class StructuredOutputService:
    """
    Schema-enforced structured output via Instructor.

    Usage::

        class Route(BaseModel):
            agent: str
            reasoning: str

        svc = StructuredOutputService(settings.llm, settings.ollama)
        result: Route = await svc.complete(messages, schema=Route)
    """

    def __init__(
        self,
        llm_settings: LLMSettings,
        ollama_settings: OllamaSettings,
    ) -> None:
        self._model = llm_settings.main_model
        self._is_anthropic = llm_settings.provider == LLMProviderEnum.ANTHROPIC
        self._client: Any = self._build_client(llm_settings, ollama_settings)

    @staticmethod
    def _build_client(
        llm_settings: LLMSettings,
        ollama_settings: OllamaSettings,
    ) -> Any:
        """Build an Instructor-patched async client for the active provider."""
        match llm_settings.provider:
            case LLMProviderEnum.OLLAMA | LLMProviderEnum.VLLM:
                from openai import AsyncOpenAI

                base_url = ollama_settings.base_url.rstrip("/") + "/v1"
                # Ollama ignores the key but the SDK requires a non-empty value.
                client = AsyncOpenAI(base_url=base_url, api_key="ollama")
                return instructor.from_openai(client, mode=instructor.Mode.JSON)
            case LLMProviderEnum.OPENAI:
                from openai import AsyncOpenAI

                return instructor.from_openai(
                    AsyncOpenAI(api_key=llm_settings.openai_api_key)
                )
            case LLMProviderEnum.ANTHROPIC:
                import anthropic

                return instructor.from_anthropic(
                    anthropic.AsyncAnthropic(api_key=llm_settings.anthropic_api_key)
                )
            case _:  # pragma: no cover - guarded by config enum
                raise LLMError(
                    f"Unsupported LLM_PROVIDER for structured output: "
                    f"{llm_settings.provider!r}"
                )

    async def complete(
        self,
        messages: list[LLMMessage],
        schema: type[ModelT],
        *,
        context: str = "",
    ) -> ModelT:
        """
        Ask the LLM to produce a validated instance of *schema*.

        Instructor injects the schema and enforces validation; retries with
        feedback are handled by ``max_retries``. Raises :class:`LLMError` if
        the model cannot produce a valid payload within the retry budget.
        """
        full_messages: list[LLMMessage] = list(messages)
        if context:
            full_messages = [
                {"role": "system", "content": f"Context: {context}"},
                *full_messages,
            ]

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "response_model": schema,
            "messages": full_messages,
            "max_retries": MAX_RETRIES,
        }
        if self._is_anthropic:
            create_kwargs["max_tokens"] = _ANTHROPIC_MAX_TOKENS

        try:
            result: ModelT = await self._client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            logger.warning(
                "structured_output_failed",
                schema=schema.__name__,
                error=str(exc),
            )
            raise LLMError(
                f"Structured output failed for schema {schema.__name__}: {exc}"
            ) from exc
        return result
