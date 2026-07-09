"""
app/agents/prompts

Versioned Jinja2 prompt templates for the agents, loaded from ``*.j2`` files in
this package. Keeping prompts in files (not string literals) makes them
diff-reviewable and lets them be iterated on without touching node code.

``render(name, **context)`` compiles and renders one template. Autoescaping is
off on purpose — these are plain-text LLM prompts, not HTML.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=False,  # plain-text prompts, not markup
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def render(template: str, /, **context: Any) -> str:
    """Render ``<template>`` (e.g. ``"supervisor.j2"``) with the given context."""
    return _env().get_template(template).render(**context).strip()
