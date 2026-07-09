"""
app/agents/nodes/route.py

route — the supervisor's single routing decision. It classifies the request into
exactly one agent (researcher / coder / direct) using *structured output* — a
validated Pydantic ``Route``, never a parsed string (this is what makes the
supervisor robust rather than demo-grade, PROJECT_CONTEXT §5.3, Phase 5).

Routing happens once. The chosen agent then owns the request; there is no second
round of supervision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agents.prompts import render
from app.agents.state import Route
from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.agents.deps import GraphDeps
    from app.agents.state import AgentState

logger = get_logger(__name__)


def make_route_node(
    deps: GraphDeps,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def route(state: AgentState) -> dict[str, Any]:
        prompt = render(
            "supervisor.j2",
            query=state.query,
            memory_context=state.memory_context,
        )
        decision = await deps.structured.complete(
            [{"role": "user", "content": prompt}], schema=Route
        )
        logger.info("node_route", agent=decision.agent, session_id=state.session_id)
        return {"route": decision}

    return route
