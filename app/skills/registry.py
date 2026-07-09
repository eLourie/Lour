"""
app/skills/registry.py

SkillRegistry — discovers skill declarations and serves ready-to-invoke Skills.

Discovery is file-driven (declarative extensibility, §1.2): every
``definitions/*.yaml`` becomes a ``SkillSpec``. If a matching Python module
exists under ``implementations/<name>.py`` and defines a ``Skill`` subclass, that
subclass is used (the override hook, e.g. review_code post-processing);
otherwise the plain ``Skill`` base drives the graph. Adding a skill is therefore
"drop a YAML file (+ optional .py) and it appears in the catalogue" — no
registry edits.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from app.core.exceptions import ConfigurationError, NotFoundError
from app.core.logging import get_logger
from app.skills.base import Skill, SkillSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)

_DEFINITIONS_DIR = Path(__file__).parent / "definitions"
_IMPLEMENTATIONS_PKG = "app.skills.implementations"


class SkillRegistry:
    """In-memory registry of skills, loaded once at startup."""

    def __init__(self, definitions_dir: Path | None = None) -> None:
        self._dir = definitions_dir or _DEFINITIONS_DIR
        self._skills: dict[str, Skill] = {}

    def load(self) -> SkillRegistry:
        """Discover and instantiate every skill. Idempotent (clears first)."""
        self._skills.clear()
        for path in sorted(self._dir.glob("*.yaml")):
            spec = self._load_spec(path)
            skill = self._instantiate(spec)
            if spec.name in self._skills:
                raise ConfigurationError(f"Duplicate skill name {spec.name!r} in {path}")
            self._skills[spec.name] = skill
        logger.info("skills_loaded", count=len(self._skills), skills=sorted(self._skills))
        return self

    @staticmethod
    def _load_spec(path: Path) -> SkillSpec:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid YAML in skill {path.name}: {exc}") from exc
        try:
            return SkillSpec.model_validate(raw)
        except Exception as exc:  # pydantic ValidationError → config problem
            raise ConfigurationError(f"Invalid skill definition {path.name}: {exc}") from exc

    @staticmethod
    def _instantiate(spec: SkillSpec) -> Skill:
        """Use a Python override subclass if one exists, else the base Skill."""
        override = _find_override(spec.name)
        return override(spec) if override is not None else Skill(spec)

    # Access

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise NotFoundError(
                f"Skill {name!r} is not registered", code="skill_not_found"
            ) from exc

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())


def _find_override(name: str) -> type[Skill] | None:
    """Return a ``Skill`` subclass from ``implementations/<name>.py``, or None."""
    try:
        module = importlib.import_module(f"{_IMPLEMENTATIONS_PKG}.{name}")
    except ModuleNotFoundError:
        return None
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Skill) and obj is not Skill and obj.__module__ == module.__name__:
            return obj
    return None
