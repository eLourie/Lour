"""
app/tools/registry.py

ToolRegistry + the ``@tool`` decorator.

Two concerns, kept separate:

  * ``@tool`` — a class decorator that validates a BaseTool subclass declares
    the required metadata (name/description/args_schema) and records the class
    for discovery. It does NOT instantiate — many tools need runtime
    dependencies (retriever, sandbox, settings), injected at construction.

  * ``ToolRegistry`` — a runtime container of *instances*. The lifespan builds
    one, registers the dependency-wired builtins (see
    ``app/tools/builtins/__init__.py``) plus any MCP-adapter tools, and exposes
    Ollama schemas for the allowed subset.

This mirrors the RAG ``default_loaders()`` pattern: declaration by decorator,
composition by an explicit factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.tools.base import BaseTool
from app.tools.schema import to_ollama_schema

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)

# Classes decorated with @tool, in declaration order. Used by the MCP server
# and introspection to enumerate the builtin catalogue without instantiating.
_DISCOVERED: list[type[BaseTool[Any]]] = []


def tool[T: type[BaseTool[Any]]](cls: T) -> T:
    """
    Register a BaseTool subclass for discovery and validate its metadata.

    Raises TypeError at import time if the class is missing required
    attributes — surfacing mistakes early rather than at first invocation.
    """
    for attr in ("name", "description", "args_schema"):
        if not getattr(cls, attr, None):
            raise TypeError(f"@tool {cls.__name__} is missing required attribute {attr!r}")
    if len(cls.name) > 30:
        raise TypeError(f"@tool {cls.name!r}: name must be ≤ 30 chars (verb_object)")
    if len(cls.description) > 1024:
        raise TypeError(f"@tool {cls.name!r}: description must be ≤ 1024 chars")
    if cls not in _DISCOVERED:
        _DISCOVERED.append(cls)
    return cls


def discovered_tool_classes() -> list[type[BaseTool[Any]]]:
    """Return the tool classes registered via ``@tool`` (declaration order)."""
    return list(_DISCOVERED)


class ToolRegistry:
    """A runtime container mapping tool name → instance."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}

    def register(self, tool_instance: BaseTool[Any], *, replace: bool = False) -> None:
        name = tool_instance.name
        if name in self._tools and not replace:
            raise ValueError(f"Tool {name!r} is already registered")
        self._tools[name] = tool_instance
        logger.debug("tool_registered", tool=name, side_effects=tool_instance.side_effects)

    def register_all(self, tools: list[BaseTool[Any]]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> BaseTool[Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise NotFoundError(
                f"Tool {name!r} is not registered",
                code="tool_not_found",
                detail={"tool": name, "available": sorted(self._tools)},
            ) from exc

    def names(self) -> set[str]:
        return set(self._tools)

    def all(self) -> list[BaseTool[Any]]:
        return list(self._tools.values())

    def to_ollama_schemas(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        """
        Return native tool schemas for the registered tools.

        ``allowed`` (from the resolved Policy) restricts the set the model can
        see; None means every registered tool.
        """
        return [
            to_ollama_schema(t)
            for name, t in self._tools.items()
            if allowed is None or name in allowed
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[BaseTool[Any]]:
        return iter(self._tools.values())
