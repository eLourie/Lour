"""
tests/unit/test_memory_scoring.py

Pure-unit tests for memory ranking math and importance caching — no I/O.
The Qdrant/Redis/LLM collaborators are replaced with tiny in-memory fakes so the
scoring logic (recency decay, alpha/beta/gamma re-ranking, cache short-circuit)
is exercised deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.core.config import MemorySettings
from app.services.memory.long_term import LongTermMemory
from app.services.memory.scoring import ImportanceJudgment, ImportanceScorer

pytestmark = pytest.mark.unit


def _make_long_term() -> LongTermMemory:
    # embedder / qdrant are unused by the pure scoring helpers under test.
    settings = MemorySettings()
    return LongTermMemory(
        qdrant=SimpleNamespace(),  # type: ignore[arg-type]
        embedder=SimpleNamespace(),  # type: ignore[arg-type]
        settings=settings,
        collection="memories",
    )


def _point(*, pid: str, score: float, importance: float, created_at: datetime, content: str):
    return SimpleNamespace(
        id=pid,
        score=score,
        payload={
            "content": content,
            "importance": importance,
            "created_at": created_at.timestamp(),
        },
    )


def test_recency_decays_by_half_life() -> None:
    lt = _make_long_term()
    now = datetime.now(tz=UTC)
    half_life_h = MemorySettings().recency_half_life_h

    fresh = lt._recency(now.timestamp(), now.timestamp())
    one_half_life = lt._recency(
        (now - timedelta(hours=half_life_h)).timestamp(), now.timestamp()
    )

    assert fresh == pytest.approx(1.0)
    assert one_half_life == pytest.approx(0.5, abs=1e-6)


def test_importance_and_recency_reorder_equal_cosine() -> None:
    """With equal cosine, a recent/important memory must outrank a stale one."""
    lt = _make_long_term()
    now = datetime.now(tz=UTC)

    stale_low = lt._to_item(
        _point(
            pid="stale",
            score=0.7,
            importance=0.1,
            created_at=now - timedelta(days=30),
            content="stale",
        ),
        now,
    )
    fresh_high = lt._to_item(
        _point(
            pid="fresh",
            score=0.7,
            importance=0.9,
            created_at=now,
            content="fresh",
        ),
        now,
    )

    assert fresh_high.score > stale_low.score
    assert fresh_high.cosine == pytest.approx(0.7)


def test_negative_cosine_is_clamped() -> None:
    lt = _make_long_term()
    now = datetime.now(tz=UTC)
    item = lt._to_item(
        _point(pid="x", score=-0.4, importance=0.5, created_at=now, content="c"), now
    )
    assert item.cosine == 0.0


class _FakeRedisHandle:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.sets = 0

    async def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        self.sets += 1
        self.store[key] = value


class _FakeStructured:
    def __init__(self, importance: float) -> None:
        self.calls = 0
        self._importance = importance

    async def complete(self, messages, schema, *, context: str = ""):
        self.calls += 1
        return ImportanceJudgment(importance=self._importance, reason="test")


async def test_importance_scorer_caches_by_hash() -> None:
    handle = _FakeRedisHandle()
    redis = SimpleNamespace(memory=handle)
    structured = _FakeStructured(importance=0.8)
    scorer = ImportanceScorer(structured, redis)  # type: ignore[arg-type]

    first = await scorer.score("Alice prefers dark mode")
    second = await scorer.score("Alice prefers dark mode")

    assert first == pytest.approx(0.8)
    assert second == pytest.approx(0.8)
    # LLM judged once; the second call is served from cache.
    assert structured.calls == 1
    assert handle.sets == 1


async def test_importance_scorer_handles_boundary_score() -> None:
    handle = _FakeRedisHandle()
    redis = SimpleNamespace(memory=handle)
    scorer = ImportanceScorer(_FakeStructured(importance=1.0), redis)  # type: ignore[arg-type]
    # Maximum salience passes through unchanged and is cached as "1.0".
    assert await scorer.score("x") == pytest.approx(1.0)
    assert handle.store[scorer._cache_key("x")] == b"1.0"
