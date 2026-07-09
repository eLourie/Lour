"""
app/skills/router.py

SkillRouter — the single text→skill classification behind ``POST /v1/skills/auto``.

Routing happens *once* (§5.3): the router picks a skill, and the skill's YAML
declares the agent — there is no second supervisor routing pass. The decision is
produced with structured output (a validated Pydantic ``SkillDecision``, never a
parsed string), then checked against the *registered* skill names. A returned
name outside the catalogue is rejected and falls back to a keyword heuristic, so
``/auto`` always resolves to a real skill (or refuses when none are registered).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.services.llm.structured import StructuredOutputService
    from app.skills.registry import SkillRegistry

logger = get_logger(__name__)

_SYSTEM = (
    "You are a router. Classify the user's request into exactly one of the "
    "available skills by returning that skill's exact name. Choose the single "
    "best fit based on each skill's stated purpose."
)


class SkillDecision(BaseModel):
    """The router's structured choice."""

    skill: str = Field(description="The exact name of the chosen skill.")
    reasoning: str = Field(default="", description="Why this skill was chosen.")


class SkillRouter:
    """Classifies free text to one registered skill via structured output."""

    def __init__(self, registry: SkillRegistry, structured: StructuredOutputService) -> None:
        self._registry = registry
        self._structured = structured

    def _catalog(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self._registry.all())

    async def classify(self, text: str) -> SkillDecision:
        """Pick the best skill for *text*. Raises ConfigurationError if none registered."""
        if len(self._registry) == 0:
            raise ConfigurationError("No skills registered — cannot route.")

        prompt = (
            f"Available skills (name: purpose):\n{self._catalog()}\n\n"
            f"User request:\n{text}\n\n"
            "Return the exact name of the single best skill."
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]
        decision = await self._structured.complete(messages, schema=SkillDecision)

        if decision.skill in self._registry:
            logger.info("skill_route", skill=decision.skill, source="llm")
            return decision

        # Model named something outside the catalogue — recover deterministically.
        fallback = self._keyword_fallback(text)
        logger.warning(
            "skill_route_fallback", returned=decision.skill, chosen=fallback, source="keyword"
        )
        return SkillDecision(
            skill=fallback,
            reasoning=f"fallback: model returned unregistered {decision.skill!r}",
        )

    def _keyword_fallback(self, text: str) -> str:
        """Pick the skill whose name/description shares the most words with *text*."""
        words = {w.strip(".,!?").lower() for w in text.split() if len(w) > 3}
        best = self._registry.names()[0]
        best_score = -1
        for skill in self._registry.all():
            haystack = f"{skill.name} {skill.description}".lower()
            score = sum(1 for w in words if w in haystack)
            if score > best_score:
                best, best_score = skill.name, score
        return best
