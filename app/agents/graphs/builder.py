"""
app/agents/graphs/builder.py

``build_graph`` — the single assembly point the app lifespan calls to get a
compiled supervisor graph, plus the helpers a request needs to drive it:

  - ``build_graph(deps, checkpointer)`` — compile the supervisor (with its
    embedded researcher/coder subgraphs) against a checkpointer.
  - ``resolve_run_policy`` — compose the effective Policy for a run
    (config defaults ← request override, most-restrictive-wins; ADR-011).
  - ``initial_state`` — the seed state dict handed to the graph at invocation.

Keeping construction here means routes and tests never wire nodes by hand — they
ask the builder for a graph and a seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agents.graphs.supervisor import compile_supervisor
from app.agents.state import AgentState, BudgetState
from app.core.policy import Policy, PolicyResolver, default_policy

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from app.agents.deps import GraphDeps


def build_graph(deps: GraphDeps, checkpointer: Any = None) -> CompiledStateGraph:
    """Compile the supervisor graph, wiring in the checkpointer if provided."""
    return compile_supervisor(deps, checkpointer=checkpointer)


def resolve_run_policy(request: Policy | None = None) -> Policy:
    """Effective policy for a run: config defaults, then any request override."""
    return PolicyResolver.resolve(defaults=default_policy(), request=request)


def initial_state(
    *,
    session_id: str,
    thread_id: str,
    query: str,
    policy: Policy | None = None,
) -> dict[str, Any]:
    """Build the seed state dict for a fresh graph invocation."""
    effective = policy if policy is not None else resolve_run_policy()
    return AgentState(
        session_id=session_id,
        thread_id=thread_id,
        query=query,
        policy=effective,
        budget=BudgetState(),
    ).model_dump()
