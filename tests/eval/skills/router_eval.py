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

pytestmark = pytest.mark.eval

# Passing bar for the DoD (§11 / Phase 6: ≥ 0.90).
ROUTING_THRESHOLD = 0.90


@dataclass(frozen=True)
class RoutingCase:
    text: str
    expected: str


# Labelled set — a few unambiguous examples per skill.
ROUTING_CASES: tuple[RoutingCase, ...] = (
    RoutingCase(
        "Research the history of the Rust programming language with sources.", "research_topic"
    ),
    RoutingCase("Do a deep dive on transformer architectures and cite papers.", "research_topic"),
    RoutingCase("Give me an in-depth report on renewable energy trends.", "research_topic"),
    RoutingCase("Review this Python function for bugs and suggest improvements.", "review_code"),
    RoutingCase("Can you check my code for correctness and run it?", "review_code"),
    RoutingCase("Look over this snippet and tell me what's wrong with it.", "review_code"),
    RoutingCase("What does my knowledge base say about our deployment process?", "answer_from_kb"),
    RoutingCase("Using only my documents, what is the retention policy?", "answer_from_kb"),
    RoutingCase("Answer from the knowledge base: who owns the billing service?", "answer_from_kb"),
    RoutingCase("Summarize this document into a few short sentences.", "summarize_document"),
    RoutingCase("Give me a detailed summary of the text below.", "summarize_document"),
    RoutingCase("Condense this article into its key points.", "summarize_document"),
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
