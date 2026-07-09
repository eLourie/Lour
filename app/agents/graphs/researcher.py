"""
app/agents/graphs/researcher.py

Researcher subgraph: ``plan → act ↻ reflect → finish``.

The node ``plan`` drafts a structured plan and seeds the transcript; ``act`` runs
one reason-act iteration (tools + LLM); ``reflect`` judges completeness. The
loop-or-finish edge is gated by *both* the reflection verdict and the
BudgetEnforcer — so a run stops when the answer is good enough, when the budget
is spent, or when a non-progressing loop is detected (ADR-007), whichever comes
first. ``finalize`` guarantees a ``final_answer`` on every exit.

Compiled without a checkpointer: it is embedded as a node in the supervisor
graph, which owns checkpointing (the subgraph inherits it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.graph import END, START, StateGraph

from app.agents.graphs._common import make_finalize_node
from app.agents.nodes.act import make_act_node
from app.agents.nodes.plan import make_plan_node
from app.agents.nodes.reflect import make_reflect_node
from app.agents.state import AgentState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from app.agents.deps import GraphDeps


def build_researcher_graph(deps: GraphDeps) -> CompiledStateGraph:
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("plan", make_plan_node(deps))
    graph.add_node("act", make_act_node(deps))
    graph.add_node("reflect", make_reflect_node(deps))
    graph.add_node("finalize", make_finalize_node())

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "reflect")

    def loop_or_finish(state: AgentState) -> Literal["act", "finalize"]:
        # Budget / loop guard wins over the reflection verdict.
        if not deps.enforcer.check(state).can_continue:
            return "finalize"
        if state.reflection is not None and state.reflection.is_complete:
            return "finalize"
        return "act"

    graph.add_conditional_edges(
        "reflect", loop_or_finish, {"act": "act", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()
