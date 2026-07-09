"""
app/services/llm/factory.py

Provider factory: reads LLM_PROVIDER and returns the right implementation.
The returned object satisfies the LLMProvider Protocol — callers never
import a concrete provider class directly.
"""

from __future__ import annotations

from app.core.config import LLMProvider as LLMProviderEnum
from app.core.config import LLMSettings, OllamaSettings
from app.core.exceptions import LLMError
from app.infra.clients.ollama import OllamaClient
from app.services.llm.base import LLMProvider
from app.services.llm.cloud_provider import build_cloud_provider
from app.services.llm.ollama_provider import OllamaProvider


def build_llm_provider(
    llm_settings: LLMSettings,
    ollama_settings: OllamaSettings,
    *,
    ollama_client: OllamaClient | None = None,
) -> LLMProvider:
    """
    Construct and return the active LLM provider.

    Changing ``LLM_PROVIDER`` in ``.env`` is sufficient — no code changes.
    Pass ``ollama_client`` to reuse the process-wide singleton (the lifespan
    owns and closes it); if omitted a fresh client is created — but then the
    caller is responsible for closing it.
    """
    match llm_settings.provider:
        case LLMProviderEnum.OLLAMA:
            client = ollama_client or OllamaClient(ollama_settings)
            ollama_provider = OllamaProvider(client, llm_settings)
            assert isinstance(ollama_provider, LLMProvider)
            return ollama_provider
        case LLMProviderEnum.ANTHROPIC | LLMProviderEnum.OPENAI:
            cloud_provider = build_cloud_provider(llm_settings)
            assert isinstance(cloud_provider, LLMProvider)
            return cloud_provider
        case _:
            raise LLMError(
                f"Unsupported LLM_PROVIDER={llm_settings.provider!r}. "
                f"Valid options: {[e.value for e in LLMProviderEnum]}"
            )
