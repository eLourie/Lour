"""
app/skills/implementations

Optional Python overrides for skills. A module named ``<skill_name>.py`` that
defines a ``Skill`` subclass is auto-discovered by the SkillRegistry and used in
place of the plain base ``Skill`` — the hook for post-processing a skill's
result beyond what the YAML declaration expresses.
"""

from __future__ import annotations
