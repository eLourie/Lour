"""
tests/eval/datasets/__init__.py

Evaluation datasets as first-class artifacts (JSONL), plus a tiny loader.

Keeping the labelled cases in versioned ``*.jsonl`` files — rather than inline
Python lists — means: (1) they read as data, diffable per row; (2) they can be
uploaded verbatim to Langfuse as a dataset and diffed between runs; (3) growing a
suite is a data edit, not a code change. Each eval module loads its dataset
through :func:`load_jsonl` and keeps only the scoring logic in code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATASETS_DIR = Path(__file__).parent


def load_jsonl(name: str) -> list[dict[str, Any]]:
    """Load a newline-delimited JSON dataset shipped in this directory.

    ``name`` is the bare filename with or without the ``.jsonl`` suffix. Blank
    lines are skipped so a trailing newline is harmless.
    """
    filename = name if name.endswith(".jsonl") else f"{name}.jsonl"
    path = _DATASETS_DIR / filename
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
