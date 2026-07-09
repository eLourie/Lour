"""
app/skills

Skills layer (PROJECT_CONTEXT §5, Phase 6): high-level business scenarios as
first-class, declarative entities.

A Skill is a YAML declaration (input/output schema, the agent it drives, its
tool allowlist and its policy) with an optional Python override for
post-processing. Skills are the *public* catalogue of what the agent can do — as
opposed to Tools, which are the private building blocks the LLM calls (§5.1).
"""

from __future__ import annotations
