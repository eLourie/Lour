"""
app/agents/nodes/plan.py

plan — the researcher subgraph's entry node. It turns the request into a short,
ordered, structured plan (a validated ``Plan``) and seeds the working message
transcript with the researcher system prompt plus the user's query, so the
following ``act`` iterations have a conversation to build on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agents.prompts import render
from app.agents.state import Plan
from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.agents.deps import GraphDeps
    from app.agents.state import AgentState

logger = get_logger(__name__)

_MAX_STEPS = 5


def make_plan_node(
    deps: GraphDeps,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def plan(state: AgentState) -> dict[str, Any]:
        system = render("researcher.j2", plan=[], memory_context=state.memory_context)
        instruction = (
            "Break the request into a short ordered list of concrete research "
            f"steps (at most {_MAX_STEPS}). Return only the steps.\n\n"
            f"Request: {state.query}"
        )
        result = await deps.structured.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
            ],
            schema=Plan,
        )
        steps = result.steps[:_MAX_STEPS]
        logger.info("node_plan", steps=len(steps), session_id=state.session_id)

        # Seed the transcript the act loop will extend: researcher system prompt
        # (now carrying the plan) + the user query.
        system_with_plan = render(
            "researcher.j2", plan=steps, memory_context=state.memory_context
        )
        return {
            "plan": steps,
            "messages": [
                {"role": "system", "content": system_with_plan},
                {"role": "user", "content": state.query},
            ],
        }

    return plan
