"""
tests/eval/skills/router_eval.py

Skill-routing eval — the Phase-6 DoD gate: classification accuracy of the
SkillRouter on a small labelled set, run against the *live* structured LLM
(real Ollama). Requires backing services.

Run standalone:
    uv run python scripts/eval_run.py --suite skill_routing
Or as an eval test:
    pytest -m eval tests/eval/skills/router_eval.py

The router only has to pick the right skill from the free text; the skill's YAML
then declares the agent (§5.3), so a single classification is all that is judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.main import create_app, lifespan
from tests.eval.datasets import load_jsonl

pytestmark = pytest.mark.eval

# Passing bar for the DoD (§11 / Phase 6: ≥ 0.90).
ROUTING_THRESHOLD = 0.90


@dataclass(frozen=True)
class RoutingCase:
    text: str
    expected: str


# Labelled set — loaded from the versioned JSONL dataset (a few unambiguous
# examples per skill). Grow the suite by editing the dataset, not this module.
ROUTING_CASES: tuple[RoutingCase, ...] = tuple(
    RoutingCase(row["text"], row["expected"]) for row in load_jsonl("skill_routing")
)


@dataclass
class RoutingResult:
    text: str
    expected: str
    predicted: str
    correct: bool


@dataclass
class RoutingReport:
    results: list[RoutingResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


async def run_router_eval() -> RoutingReport:
    """Classify every labelled case through the live SkillRouter."""
    app = create_app()
    report = RoutingReport()
    async with lifespan(app):
        router = app.state.skill_router
        for case in ROUTING_CASES:
            decision = await router.classify(case.text)
            report.results.append(
                RoutingResult(
                    text=case.text,
                    expected=case.expected,
                    predicted=decision.skill,
                    correct=decision.skill == case.expected,
                )
            )
    return report


def print_report(report: RoutingReport) -> None:
    print("\n=== Skill routing eval ===")
    for r in report.results:
        mark = "✓" if r.correct else "✗"
        print(f"  {mark} exp={r.expected:20s} got={r.predicted:20s} | {r.text[:50]}")
    print(f"\naccuracy: {report.accuracy:.0%} ({report.correct}/{report.total})")


@pytest.mark.asyncio
async def test_router_accuracy() -> None:
    report = await run_router_eval()
    print_report(report)
    assert report.accuracy >= ROUTING_THRESHOLD, (
        f"routing accuracy {report.accuracy:.0%} below {ROUTING_THRESHOLD:.0%}"
    )
