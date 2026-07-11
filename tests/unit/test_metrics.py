"""
tests/unit/test_metrics.py

Unit coverage for the in-process metrics registry (app/core/metrics.py):
aggregation math, cost pricing by longest-prefix match, and snapshot shape.
Pure — no I/O.
"""

from __future__ import annotations

import pytest

from app.core.metrics import MetricsRegistry, _price_for, get_metrics

pytestmark = pytest.mark.unit


@pytest.fixture
def reg() -> MetricsRegistry:
    return MetricsRegistry()


def test_latency_aggregates_avg_min_max(reg: MetricsRegistry) -> None:
    for ms in (10.0, 20.0, 30.0):
        reg.record_latency("rag_retrieval", ms)
    snap = reg.snapshot()["latency_by_layer"]["rag_retrieval"]
    assert snap["count"] == 3
    assert snap["avg_ms"] == 20.0
    assert snap["min_ms"] == 10.0
    assert snap["max_ms"] == 30.0


def test_tool_success_rate(reg: MetricsRegistry) -> None:
    reg.record_tool("web_search", ok=True)
    reg.record_tool("web_search", ok=True)
    reg.record_tool("web_search", ok=False)
    tools = reg.snapshot()["tools"]
    assert tools["by_name"]["web_search"] == {"calls": 3, "ok": 2, "success_rate": 0.667}
    assert tools["overall_success_rate"] == 0.667


def test_retrieval_recall_running_average(reg: MetricsRegistry) -> None:
    reg.record_retrieval(1.0)
    reg.record_retrieval(0.0)
    reg.record_retrieval(0.5)
    retrieval = reg.snapshot()["retrieval"]
    assert retrieval["observations"] == 3
    assert retrieval["avg_recall"] == 0.5


def test_cost_uses_price_table_and_accumulates(reg: MetricsRegistry) -> None:
    # 1M input + 1M output @ (3.00, 15.00) = 18.00 USD
    reg.record_llm("anthropic", "claude-3-5-sonnet-20241022", 1_000_000, 1_000_000)
    cost = reg.snapshot()["cost"]
    entry = cost["by_model"]["claude-3-5-sonnet-20241022"]
    assert entry["calls"] == 1
    assert entry["usd"] == 18.0
    assert cost["total_usd"] == 18.0


def test_price_longest_prefix_wins() -> None:
    # gpt-4o-mini must not be captured by the shorter gpt-4o prefix.
    assert _price_for("gpt-4o-mini") == (0.15, 0.60)
    assert _price_for("gpt-4o-2024-08-06") == (2.50, 10.00)


def test_local_model_is_free() -> None:
    assert _price_for("qwen3:14b") == (0.0, 0.0)


def test_reset_clears_all(reg: MetricsRegistry) -> None:
    reg.record_latency("x", 1.0)
    reg.record_tool("t", ok=True)
    reg.record_retrieval(1.0)
    reg.record_llm("openai", "gpt-4o", 100, 100)
    reg.reset()
    snap = reg.snapshot()
    assert snap["latency_by_layer"] == {}
    assert snap["tools"]["by_name"] == {}
    assert snap["retrieval"]["observations"] == 0
    assert snap["cost"]["by_model"] == {}


def test_get_metrics_is_singleton() -> None:
    assert get_metrics() is get_metrics()
