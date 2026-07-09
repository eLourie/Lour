"""
app/skills/implementations/review_code.py

Python override for the ``review_code`` skill: post-processes the base run's
result to surface a little review-specific structure the plain YAML skill can't.

It records whether the reviewed code was actually executed in the sandbox
(``code_exec`` appears in the tool ledger) and the declared language into
``metadata`` — cheap signal a UI or eval can key on without parsing the prose.
The base ``Skill`` does all the graph driving; this only shapes the result.
"""

from __future__ import annotations

from typing import Any

from app.skills.base import Skill, SkillResult


class ReviewCodeSkill(Skill):
    """review_code with result post-processing."""

    async def postprocess(self, result: SkillResult, state: dict[str, Any]) -> SkillResult:
        result.metadata.update(
            {
                # Did the reviewer actually run the code, or only read it?
                "executed": "code_exec" in result.tools_called,
                "tools_used": len(result.tools_called),
            }
        )
        return result
