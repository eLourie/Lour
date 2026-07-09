"""
tests/integration/test_checkpoint_resume.py

Integration test for checkpoint durability — requires PostgreSQL, Redis, Qdrant
and Ollama (the supervisor graph is wired exactly as in production via lifespan).

Run with: pytest -m integration tests/integration/test_checkpoint_resume.py

The DoD is "after a kill the session resumes from its checkpoint". We prove the
mechanism deterministically: a run's state, persisted by the app's graph, is read
back by a *brand-new* AsyncPostgresSaver on an independent connection — i.e. what
a freshly restarted process would do — and it recovers the finished run.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI

from app.agents.checkpointing import CheckpointerManager
from app.agents.graphs.builder import initial_state
from app.core.config import get_settings
from app.main import create_app, lifespan

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def app_ctx() -> AsyncIterator[FastAPI]:
    app = create_app()
    async with lifespan(app):
        yield app


async def test_state_persists_and_is_recoverable(app_ctx: FastAPI) -> None:
    graph = app_ctx.state.agent_graph
    thread = f"resume-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread}}

    seed = initial_state(
        session_id=thread,
        thread_id=thread,
        query="Reply with exactly the single word: PONG",
    )
    result = await graph.ainvoke(seed, config=config)
    assert result["final_answer"], "run should produce a final answer"
    assert result["finished"] is True

    # Simulate a process restart: a fresh saver on a new connection reads the
    # same thread's checkpoint and recovers the persisted state.
    fresh = CheckpointerManager(get_settings().postgres)
    await fresh.start()
    try:
        tup = await fresh.saver.aget_tuple(config)
        assert tup is not None, "checkpoint for the thread must be persisted"
        values = tup.checkpoint["channel_values"]
        assert values.get("finished") is True
        assert values.get("final_answer"), "final answer must survive the 'restart'"
    finally:
        await fresh.aclose()


async def test_unknown_thread_has_no_checkpoint(app_ctx: FastAPI) -> None:
    graph = app_ctx.state.agent_graph
    config = {"configurable": {"thread_id": f"ghost-{uuid.uuid4().hex[:8]}"}}
    snapshot = await graph.aget_state(config)
    # No run ever touched this thread → empty state, nothing pending.
    assert not snapshot.next
    assert not snapshot.values
