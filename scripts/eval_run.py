#!/usr/bin/env python
"""
scripts/eval_run.py

Evaluation runner. Phase 2 wires the RAG suite; later phases (agents, skill
routing) extend the dispatch table.

Usage:
    uv run python scripts/eval_run.py --suite rag
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from tests.eval.agents.benchmark import (
    SUCCESS_THRESHOLD,
    print_report,
    run_agent_benchmark,
)
from tests.eval.rag.ragas_suite import _print_report, run_rag_eval
from tests.eval.skills.router_eval import (
    ROUTING_THRESHOLD,
    run_router_eval,
)
from tests.eval.skills.router_eval import (
    print_report as print_routing_report,
)


async def _run_rag() -> int:
    metrics = await run_rag_eval()
    _print_report(metrics)
    hybrid = metrics["hybrid"].recall_at_k
    dense = metrics["dense"].recall_at_k
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation suites.")
    parser.add_argument("--suite", choices=["rag", "agents", "skill_routing"], required=True)
    args = parser.parse_args()

    if args.suite == "rag":
        return asyncio.run(_run_rag())
    if args.suite == "agents":
        return asyncio.run(_run_agents())
    if args.suite == "skill_routing":
        return asyncio.run(_run_skill_routing())
    return 1  # pragma: no cover — argparse guards choices


if __name__ == "__main__":
    sys.exit(main())
