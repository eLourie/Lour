"""
app/core/metrics.py

Lightweight, in-process metrics registry — the offline-baseline counterpart to
Langfuse (§3.4, §4 cross-cutting). structlog already emits per-event JSON lines;
this module *aggregates* them into a snapshot that the admin ``/metrics`` route
can serve and that eval runs can print, without standing up Prometheus.

Four families, matching the Phase-8 DoD (latency by layer, tool success rate,
retrieval recall, cost tracking for the cloud tier):

    record_latency(layer, ms)                  → per-layer LatencyStat
    record_tool(name, ok)                      → per-tool success counter
    record_retrieval(recall)                   → running recall aggregate
    record_llm(provider, model, in_, out)      → token + USD cost by model

Design notes:
  * Bounded memory: everything keys on a small, stable label set (layer name,
    tool name, model) and aggregates in place — no unbounded per-call lists.
  * Never throws: recording is best-effort telemetry and must not break a
    request. Callers wrap nothing; the methods themselves swallow nothing but
    also do no work that can fail on well-typed input.
  * Thread-safe: the sandbox and other CPU-bound paths run under
    ``asyncio.to_thread``, so a plain lock guards the mutations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

# Approximate cloud pricing, USD per 1M tokens (input, output). Local Ollama is
# free (0.0). Kept deliberately small and marked approximate — cost tracking is
# for order-of-magnitude budget awareness on the reliability tier, not billing.
# Matched by longest known prefix so dated model ids (e.g. the -20241022 suffix)
# still resolve.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "deepseek": (0.27, 1.10),
}


def _price_for(model: str) -> tuple[float, float]:
    """Return (input, output) USD per 1M tokens for a model id, or (0, 0)."""
    name = model.lower()
    best: tuple[float, float] | None = None
    best_len = -1
    for prefix, price in _PRICE_PER_MTOK.items():
        if name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = price, len(prefix)
    return best if best is not None else (0.0, 0.0)


@dataclass
class LatencyStat:
    """Aggregate latency for one layer/span (no per-call retention)."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def observe(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.count else 0.0,
            "max_ms": round(self.max_ms, 2),
        }


@dataclass
class ToolStat:
    """Call / success counters for one tool."""

    calls: int = 0
    ok: int = 0

    @property
    def success_rate(self) -> float:
        return self.ok / self.calls if self.calls else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {"calls": self.calls, "ok": self.ok, "success_rate": round(self.success_rate, 3)}


@dataclass
class CostStat:
    """Token and USD accounting for one model id."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usd": round(self.usd, 6),
        }


@dataclass
class _RetrievalAgg:
    """Running recall aggregate (count + sum), no per-observation list."""

    count: int = 0
    total: float = 0.0

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class MetricsRegistry:
    """Process-global aggregator. Reach it via :func:`get_metrics`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latency: dict[str, LatencyStat] = {}
        self._tools: dict[str, ToolStat] = {}
        self._cost: dict[str, CostStat] = {}
        self._retrieval = _RetrievalAgg()

    # ── recording ──────────────────────────────────────────────────────────

    def record_latency(self, layer: str, ms: float) -> None:
        with self._lock:
            self._latency.setdefault(layer, LatencyStat()).observe(ms)

    def record_tool(self, name: str, ok: bool) -> None:
        with self._lock:
            stat = self._tools.setdefault(name, ToolStat())
            stat.calls += 1
            if ok:
                stat.ok += 1

    def record_retrieval(self, recall: float) -> None:
        with self._lock:
            self._retrieval.count += 1
            self._retrieval.total += recall

    def record_llm(
        self, provider: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        in_price, out_price = _price_for(model)
        usd = (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price
        with self._lock:
            stat = self._cost.setdefault(model, CostStat())
            stat.calls += 1
            stat.prompt_tokens += prompt_tokens
            stat.completion_tokens += completion_tokens
            stat.usd += usd

    # ── reporting ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """A JSON-serialisable view for the admin route and eval reports."""
        with self._lock:
            total_calls = sum(s.calls for s in self._tools.values())
            total_ok = sum(s.ok for s in self._tools.values())
            return {
                "latency_by_layer": {k: v.as_dict() for k, v in sorted(self._latency.items())},
                "tools": {
                    "by_name": {k: v.as_dict() for k, v in sorted(self._tools.items())},
                    "overall_success_rate": round(total_ok / total_calls, 3)
                    if total_calls
                    else 0.0,
                },
                "retrieval": {
                    "observations": self._retrieval.count,
                    "avg_recall": round(self._retrieval.avg, 3),
                },
                "cost": {
                    "by_model": {k: v.as_dict() for k, v in sorted(self._cost.items())},
                    "total_usd": round(sum(s.usd for s in self._cost.values()), 6),
                },
            }

    def reset(self) -> None:
        """Clear all counters (used by tests)."""
        with self._lock:
            self._latency.clear()
            self._tools.clear()
            self._cost.clear()
            self._retrieval = _RetrievalAgg()


# Module-level singleton. Metrics are process-scoped, like the structlog config.
_registry = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    """Return the process-global metrics registry."""
    return _registry
