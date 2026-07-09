"""
app/agents/deps.py

GraphDeps — the bundle of services the graph nodes need, injected once when the
graph is built. Keeping it in its own module (rather than in builder.py) breaks
the import cycle: nodes import ``GraphDeps`` for typing, and the builder imports
the nodes.

Nodes are plain async callables produced by ``make_*_node(deps)`` factories that
close over this bundle — LangGraph has no DI of its own, so the closure *is* the
injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.enforcement.budget import BudgetEnforcer
    from app.core.config import Settings
    from app.services.llm.base import LLMProvider
    from app.services.llm.structured import StructuredOutputService
    from app.services.memory.base import MemoryManager
    from app.tools.gate import ToolGate
    from app.tools.registry import ToolRegistry


@dataclass(frozen=True)
class GraphDeps:
    """Services shared by every node in the supervisor graph."""

    llm: LLMProvider
    structured: StructuredOutputService
    tool_registry: ToolRegistry
    tool_gate: ToolGate
    memory: MemoryManager
    enforcer: BudgetEnforcer
    settings: Settings
