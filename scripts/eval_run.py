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

from tests.eval.rag.ragas_suite import _print_report, run_rag_eval


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation suites.")
    parser.add_argument("--suite", choices=["rag"], required=True)
    args = parser.parse_args()

    if args.suite == "rag":
        return asyncio.run(_run_rag())
    return 1  # pragma: no cover — argparse guards choices


if __name__ == "__main__":
    sys.exit(main())
