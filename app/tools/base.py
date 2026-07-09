"""
app/tools/base.py

Tool contract: BaseTool (ABC) + ToolResult.

A Tool is a low-level, stateless function the LLM invokes (``web_search``,
``code_exec``, ...). Tools do NOT know about the LLM, agent state or memory —
they take typed args and return a uniform ``ToolResult``.

Design rules (PROJECT_CONTEXT §5, Phase 3):
  - name       — ``verb_object`` ≤ 30 chars, matches ToolRegistry key.
  - description — starts with a verb, ≤ 1024 chars, mentions when NOT to use it.
  - args_schema — a Pydantic model; arguments are strictly typed.
  - execute()  — returns ToolResult{ok, data, error, metadata} and never raises
                 for expected failures; ``run()`` wraps it to guarantee this.
  - side_effects — True for tools that mutate external state (skip caching,
                 eligible for HITL approval via ToolGate).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.core.exceptions import ToolError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Re-exported so callers can `from app.tools.base import ToolError`.
__all__ = ["BaseTool", "ToolError", "ToolResult"]


class ToolResult(BaseModel):
    """Uniform result envelope returned by every tool."""

    ok: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(cls, data: Any = None, **metadata: Any) -> ToolResult:
        return cls(ok=True, data=data, metadata=metadata)

    @classmethod
    def failure(cls, error: str, **metadata: Any) -> ToolResult:
        return cls(ok=False, error=error, metadata=metadata)


class BaseTool[ArgsT: BaseModel](ABC):
    """
    Abstract base for all tools.

    Builtin subclasses set ``name``, ``description`` and ``args_schema`` in the
    class body; dynamic tools (e.g. the MCP adapter) set them per-instance in
    ``__init__`` — hence plain attributes, not ClassVars. Call ``run(raw_args)``
    from the orchestration layer: it validates arguments against the schema and
    turns any unexpected exception into a failed ToolResult so a single bad tool
    call never crashes the agent loop.
    """

    name: str
    description: str
    args_schema: type[BaseModel]
    # Mutates external state → not cacheable, may require HITL approval.
    side_effects: bool = False

    @abstractmethod
    async def execute(self, args: ArgsT) -> ToolResult:
        """Run the tool with validated arguments."""
        ...

    def ollama_parameters(self) -> dict[str, Any]:
        """
        JSON Schema for this tool's arguments (native tool-calling).

        Default derives it from ``args_schema``. The MCP adapter overrides this
        to surface the remote server's own inputSchema verbatim.
        """
        return self.args_schema.model_json_schema()

    async def run(self, raw_args: dict[str, Any] | None = None) -> ToolResult:
        """
        Validate ``raw_args`` against ``args_schema`` and execute.

        Guarantees a ToolResult is returned: validation errors become a failed
        result (the LLM can retry with corrected args), and unexpected
        exceptions are caught and logged rather than propagated.
        """
        try:
            args = self.args_schema.model_validate(raw_args or {})
        except ValidationError as exc:
            logger.info("tool_args_invalid", tool=self.name, errors=exc.error_count())
            return ToolResult.failure(
                f"Invalid arguments for tool {self.name!r}: {exc}",
                error_type="validation",
            )

        try:
            return await self.execute(args)  # type: ignore[arg-type]
        except ToolError as exc:
            logger.warning("tool_error", tool=self.name, error=str(exc))
            return ToolResult.failure(str(exc), error_type="tool_error")
        except Exception as exc:
            logger.exception("tool_unhandled", tool=self.name)
            return ToolResult.failure(
                f"Tool {self.name!r} failed: {exc}", error_type="unhandled"
            )

    def __repr__(self) -> str:
        return f"<Tool {self.name} side_effects={self.side_effects}>"
