"""
tests/integration/test_memory_recall.py

Integration test for cross-session memory recall — requires PostgreSQL, Redis,
Qdrant and Ollama (dense embeddings).

Run with: pytest -m integration tests/integration/test_memory_recall.py

The app lifespan is entered so app.state.memory is wired exactly as in
production (short-term Redis, long-term Qdrant, episodic Postgres behind the
MemoryManager facade).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from app.core.config import MemorySettings, get_settings
from app.infra.clients.redis import RedisClient
from app.main import create_app, lifespan
from app.services.llm.base import LLMResponse
from app.services.memory.base import MemoryManager
from app.services.memory.short_term import ShortTermMemory

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def memory() -> AsyncIterator[MemoryManager]:
    app = create_app()
    async with lifespan(app):
        yield app.state.memory


class _StubLLM:
    """Deterministic stand-in for the summariser LLM (keeps the test fast)."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="ROLLING SUMMARY", model="stub")


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[RedisClient]:
    client = RedisClient(get_settings().redis)
    yield client
    await client.aclose()


async def test_short_term_window_evicts_and_summarises(redis_client: RedisClient) -> None:
    """Overflow beyond the window evicts the oldest turns into a rolling summary."""
    settings = MemorySettings(short_term_window=3)
    stub = _StubLLM()
    stm = ShortTermMemory(redis_client, stub, settings)  # type: ignore[arg-type]
    session_id = f"evict-{uuid.uuid4().hex[:8]}"

    try:
        for i in range(5):
            await stm.append(session_id, "user", f"message number {i}")

        window = await stm.get_window(session_id)
        # Only the last `window` turns are kept verbatim.
        assert [t.content for t in window] == [
            "message number 2",
            "message number 3",
            "message number 4",
        ]
        # The evicted tail was folded into a summary via the LLM.
        assert stub.calls >= 1
        assert await stm.get_summary(session_id) == "ROLLING SUMMARY"
    finally:
        await stm.clear(session_id)


async def test_long_term_fact_recalled_across_sessions(memory: MemoryManager) -> None:
    marker = uuid.uuid4().hex[:8]
    fact = f"The Zephyr-{marker} service is written in the Rust programming language."

    # Session A learns a durable fact.
    await memory.remember_fact(fact, importance=0.9, session_id=f"session-A-{marker}")

    # Session B — a *different* session — asks a related question.
    ctx = await memory.recall(
        f"session-B-{marker}",
        f"What language is the Zephyr-{marker} service written in?",
    )

    assert ctx.long_term, "expected long-term memory to surface a cross-session fact"
    assert any(f"Zephyr-{marker}" in item.content for item in ctx.long_term)
    top = ctx.long_term[0]
    assert 0.0 <= top.importance <= 1.0
    assert top.score > 0.0


async def test_short_term_window_is_session_scoped(memory: MemoryManager) -> None:
    marker = uuid.uuid4().hex[:8]
    session_a = f"stm-A-{marker}"
    session_b = f"stm-B-{marker}"

    await memory.write(session_a, "user", f"My favourite colour is teal-{marker}.")

    # Session A sees its own working window …
    ctx_a = await memory.recall(session_a, "colour")
    assert any(f"teal-{marker}" in turn.content for turn in ctx_a.recent)

    # … while a different session's window does not.
    ctx_b = await memory.recall(session_b, "colour")
    assert all(f"teal-{marker}" not in turn.content for turn in ctx_b.recent)

    await memory.clear_session(session_a)
