#!/usr/bin/env python
"""
scripts/eval_run.py

Evaluation runner — the entry point behind ``make eval``.

With no ``--suite`` (or ``--suite all``) it orchestrates the *full* suite:
RAG retrieval + quality, agent benchmark, and skill routing, then prints the
observability snapshot (layer latency, tool success, retrieval recall, cost).
Each suite is isolated so one failure still yields a full report; the process
exits non-zero if any gate fails or errors.

Tiers (§ADR-002): the suites run against whatever ``LLM_PROVIDER`` is configured.
The default is the local Ollama tier; set ``LLM_PROVIDER=anthropic`` (or openai)
in ``.env`` to run the same suite against the cloud reliability tier.

Usage:
    uv run python scripts/eval_run.py                 # full suite (all)
    uv run python scripts/eval_run.py --suite rag     # a single suite
    make eval                                          # == full suite
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from tests.eval.agents.benchmark import (
    SUCCESS_THRESHOLD,
    print_report,
    run_agent_benchmark,
)
from tests.eval.rag.ragas_suite import (
    _print_quality,
    _print_report,
    run_rag_eval,
    run_rag_quality_eval,
)
from tests.eval.skills.router_eval import (
    ROUTING_THRESHOLD,
    run_router_eval,
)
from tests.eval.skills.router_eval import (
    print_report as print_routing_report,
)

from app.core.metrics import get_metrics


async def _run_rag() -> int:
    metrics = await run_rag_eval()
    _print_report(metrics)
    hybrid = metrics["hybrid"].recall_at_k
    dense = metrics["dense"].recall_at_k
    # Best-effort LLM-judged quality (skips cleanly without the eval extra).
    _print_quality(await run_rag_quality_eval())
    # Non-regression check (see ragas_suite docstring on bge-m3 saturation).
    if hybrid >= dense:
        print(f"✓ hybrid recall@k ({hybrid:.3f}) >= pure dense ({dense:.3f})")
        return 0
    print(f"✗ hybrid recall@k ({hybrid:.3f}) < pure dense ({dense:.3f})")
    return 1


async def _run_agents() -> int:
    report = await run_agent_benchmark()
    print_report(report)
    if report.success_rate >= SUCCESS_THRESHOLD:
        print(f"✓ agent success rate {report.success_rate:.0%} >= {SUCCESS_THRESHOLD:.0%}")
        return 0
    print(f"✗ agent success rate {report.success_rate:.0%} < {SUCCESS_THRESHOLD:.0%}")
    return 1


async def _run_skill_routing() -> int:
    report = await run_router_eval()
    print_routing_report(report)
    if report.accuracy >= ROUTING_THRESHOLD:
        print(f"✓ routing accuracy {report.accuracy:.0%} >= {ROUTING_THRESHOLD:.0%}")
        return 0
    print(f"✗ routing accuracy {report.accuracy:.0%} < {ROUTING_THRESHOLD:.0%}")
    return 1


# Dispatch table for single-suite runs.
_SUITES = {
    "rag": _run_rag,
    "agents": _run_agents,
    "skill_routing": _run_skill_routing,
}


def _print_metrics_snapshot() -> None:
    print("\n=== Observability snapshot (app/core/metrics) ===")
    print(json.dumps(get_metrics().snapshot(), indent=2, sort_keys=True))


async def _run_all() -> int:
    """Run every suite, isolating failures, then print the metrics snapshot."""
    outcomes: dict[str, str] = {}
    for name, runner in _SUITES.items():
        print(f"\n{'=' * 70}\n▶ suite: {name}\n{'=' * 70}")
        try:
            code = await runner()
            outcomes[name] = "pass" if code == 0 else "fail"
        except Exception as exc:  # a suite that cannot even run is an error, not a pass
            outcomes[name] = f"error: {exc}"
            print(f"✗ suite {name} errored: {exc}")

    _print_metrics_snapshot()

    print("\n=== Eval summary ===")
    for name, status in outcomes.items():
        mark = "✓" if status == "pass" else "✗"
        print(f"  {mark} {name:<14} {status}")

    return 0 if all(v == "pass" for v in outcomes.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation suites.")
    parser.add_argument(
        "--suite",
        choices=["all", "rag", "agents", "skill_routing"],
        default="all",
        help="Which suite to run (default: all).",
    )
    args = parser.parse_args()

    if args.suite == "all":
        return asyncio.run(_run_all())
    return asyncio.run(_SUITES[args.suite]())


if __name__ == "__main__":
    sys.exit(main())
