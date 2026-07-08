"""
tests/integration/test_llm_ollama.py

Integration tests for OllamaProvider — require a running Ollama instance.
Run with: pytest -m integration tests/integration/test_llm_ollama.py

These tests cover:
  1. Non-streaming chat completion
  2. Token streaming
  3. Structured output via StructuredOutputService
  4. Dense embeddings
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.core.config import get_settings
from app.infra.clients.ollama import OllamaClient
from app.services.llm.factory import build_llm_provider
from app.services.llm.structured import StructuredOutputService

pytestmark = pytest.mark.integration


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def ollama_client(settings):
    return OllamaClient(settings.ollama)


@pytest.fixture
def llm_provider(settings, ollama_client):
    from app.services.llm.ollama_provider import OllamaProvider
    return OllamaProvider(ollama_client, settings.llm)


@pytest.mark.asyncio
async def test_chat_returns_content(llm_provider):
    response = await llm_provider.chat(
        [{"role": "user", "content": "Say exactly: hello"}]
    )
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_stream_yields_tokens(llm_provider):
    tokens: list[str] = []
    async for chunk in llm_provider.stream(
        [{"role": "user", "content": "Count to 3"}]
    ):
        tokens.append(chunk.token)
        if chunk.done:
            break
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_structured_output(llm_provider):
    class Greeting(BaseModel):
        message: str
        language: str

    svc = StructuredOutputService(llm_provider)
    result = await svc.complete(
        [{"role": "user", "content": "Greet me in English"}],
        schema=Greeting,
    )
    assert isinstance(result.message, str)
    assert isinstance(result.language, str)


@pytest.mark.asyncio
async def test_embed_returns_vectors(llm_provider):
    vectors = await llm_provider.embed(["hello world", "test sentence"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024  # bge-m3 dimensionality