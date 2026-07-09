"""
tests/integration/test_consolidation.py

Integration test for background consolidation — requires PostgreSQL, Redis,
Qdrant and Ollama (LLM fact extraction + importance scoring + embeddings).

Run with: pytest -m integration tests/integration/test_consolidation.py

Consolidation is exercised directly via ``consolidate_session`` (the same code
the APScheduler job runs) so the test is deterministic and does not wait on the
interval trigger.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI

from app.main import create_app, lifespan

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def app() -> AsyncIterator[FastAPI]:
    application = create_app()
    async with lifespan(application):
        yield application


async def test_consolidation_extracts_and_dedupes(app: FastAPI) -> None:
    memory = app.state.memory
    consolidation = app.state.consolidation
    marker = uuid.uuid4().hex[:8]
    session_id = f"consolidate-{marker}"

    # A short session containing one clearly durable fact.
    turns = [
        ("user", f"Hi, I'm working on a project codenamed Nimbus-{marker}."),
        ("assistant", "Nice to meet you. Tell me about it."),
        (
            "user",
            f"Nimbus-{marker} is a weather forecasting service and it must be "
            f"written in Go because the team standardised on Go.",
        ),
        ("assistant", "Understood — Go it is."),
    ]
    for role, content in turns:
        await memory.write(session_id, role, content)

    # First consolidation should extract and persist at least one fact.
    report = await consolidation.consolidate_session(session_id)
    assert report.extracted >= 1
    assert report.written >= 1, report.model_dump()

    # The distilled fact is now recallable from a fresh session.
    ctx = await memory.recall(f"other-{marker}", f"What language is Nimbus-{marker} in?")
    assert any(f"Nimbus-{marker}" in item.content for item in ctx.long_term)

    # Re-running consolidation dedupes — nothing new is written.
    report2 = await consolidation.consolidate_session(session_id)
    assert report2.written == 0
    assert report2.skipped_duplicate >= 1

    # The episodic ledger recorded the consolidation event.
    timeline = await memory._episodic.timeline(session_id=session_id, limit=50)
    assert any(entry.event_type == "consolidation" for entry in timeline)
    assert any(entry.event_type == "message" for entry in timeline)

    await memory.clear_session(session_id)
