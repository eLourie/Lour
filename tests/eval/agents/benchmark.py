"""
tests/eval/agents/benchmark.py

Agent benchmark — the Phase-5 DoD gate: end-to-end success rate on a small set of
reference tasks, run against the *live* supervisor graph (real Ollama, real
sandbox). Requires all backing services.

Run standalone:
    uv run python scripts/eval_run.py --suite agents
Or as an eval test:
    pytest -m eval tests/eval/agents/benchmark.py

Success is judged on the produced answer (an expected keyword must appear, or —
for open tasks — the answer must be non-empty). Routing is reported alongside but
does not gate, since more than one agent can legitimately answer some tasks.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from app.agents.graphs.builder import initial_state
from app.main import create_app, lifespan

pytestmark = pytest.mark.eval

# Passing bar for the DoD.
SUCCESS_THRESHOLD = 0.80


@dataclass(frozen=True)
class Task:
    id: str
    query: str
    expect_keywords: tuple[str, ...] = ()
    expect_agent: str | None = None


@dataclass
class TaskResult:
    id: str
    routed: str | None
    answer: str
    passed: bool
    routed_as_expected: bool


@dataclass
class BenchmarkReport:
    results: list[TaskResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def routing_accuracy(self) -> float:
        judged = [r for r in self.results if r.routed_as_expected is not None]
        if not judged:
            return 0.0
        return sum(1 for r in judged if r.routed_as_expected) / len(judged)


BENCHMARK_TASKS: tuple[Task, ...] = (
    Task(
        id="code-sum",
        query=(
            "Use Python to compute the sum of all integers from 1 to 100 "
            "and print only the number."
        ),
        expect_keywords=("5050",),
        expect_agent="coder",
    ),
    Task(
        id="code-reverse",
        query="Write and run Python that reverses the string 'hello' and prints the result.",
        expect_keywords=("olleh",),
        expect_agent="coder",
    ),
    Task(
        id="direct-capital",
        query="What is the capital of France? Answer with just the city name.",
        expect_keywords=("Paris",),
    ),
    Task(
        id="direct-echo",
        query="Reply with exactly this word and nothing else: BANANA",
        expect_keywords=("BANANA",),
    ),
)


def _judge(task: Task, answer: str) -> bool:
    text = answer.strip()
    if not text:
        return False
    if task.expect_keywords:
        return any(kw.lower() in text.lower() for kw in task.expect_keywords)
    return True


async def run_agent_benchmark() -> BenchmarkReport:
    """Run every benchmark task through a live supervisor graph."""
    app = create_app()
    report = BenchmarkReport()
    async with lifespan(app):
        graph = app.state.agent_graph
        for task in BENCHMARK_TASKS:
            thread = f"bench-{task.id}-{uuid.uuid4().hex[:6]}"
            seed = initial_state(session_id=thread, thread_id=thread, query=task.query)
            state = await graph.ainvoke(seed, config={"configurable": {"thread_id": thread}})

            route = state.get("route")
            routed = getattr(route, "agent", None)
            answer = state.get("final_answer") or ""
            passed = _judge(task, answer)
            routed_ok = task.expect_agent is None or routed == task.expect_agent
            report.results.append(
                TaskResult(
                    id=task.id,
                    routed=routed,
                    answer=answer,
                    passed=passed,
                    routed_as_expected=routed_ok,
                )
            )
    return report


def print_report(report: BenchmarkReport) -> None:
    print("\n=== Agent benchmark ===")
    for r in report.results:
        mark = "✓" if r.passed else "✗"
        preview = r.answer.replace("\n", " ")[:80]
        print(f"  {mark} {r.id:16s} route={r.routed or '-':10s} | {preview}")
    print(
        f"\nsuccess rate: {report.success_rate:.0%} ({report.passed}/{report.total})  "
        f"routing accuracy: {report.routing_accuracy:.0%}"
    )


@pytest.mark.asyncio
async def test_agent_benchmark_success_rate() -> None:
    report = await run_agent_benchmark()
    print_report(report)
    assert report.success_rate >= SUCCESS_THRESHOLD, (
        f"agent success rate {report.success_rate:.0%} below {SUCCESS_THRESHOLD:.0%}"
    )
