"""
app/agents/nodes/act.py

act — one reason-act iteration: the LLM sees the transcript and the allowed tool
schemas, decides (via native function calling) whether to call tools, and this
node executes them and folds the results back into the transcript. The subgraph
loops this node until the model answers without calling a tool (or the budget /
loop guard forces a stop).

Graceful degradation on a 14B model (ADR-007) lives here:
  - tool-name validation — a call to an unregistered tool becomes a corrective
    tool message ("that tool does not exist, choose a valid one"), not a crash;
    the model retries with feedback on the next iteration.
  - ToolGate enforcement — the resolved policy's allowlist is applied, and
    side-effecting tools flagged for approval trigger a single HITL interrupt
    *before any tool in the turn runs*, so a resume never double-executes a tool.
  - budget accounting — tokens (from the chat response), tool calls and the
    iteration counter advance ``state.budget`` for the BudgetEnforcer to read.

Tool/token progress is emitted as LangGraph custom stream events so the SSE
layer can surface them live.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from app.agents.state import PendingApproval, ToolCallRecord
from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.agents.deps import GraphDeps
    from app.agents.state import AgentState
    from app.services.llm.base import LLMMessage
    from app.tools.base import ToolResult

logger = get_logger(__name__)

# Deterministic tool-calling — temperature 0 makes the model's tool choices
# stable, which also keeps an interrupted-and-resumed act iteration consistent.
_ACT_OPTIONS = {"temperature": 0.0}

# Cap a tool result folded back into the transcript so a huge payload cannot
# blow the context window.
_MAX_TOOL_CONTENT = 8000


def _emit(event: str, **data: Any) -> None:
    """Best-effort custom stream event (no-op outside a streaming run)."""
    try:
        writer = get_stream_writer()
    except Exception:  # pragma: no cover - defensive; writer absent off-stream
        return
    writer({"event": event, **data})


def _assistant_message(content: str, tool_calls: list[dict[str, Any]]) -> LLMMessage:
    """Rebuild the assistant turn in Ollama's message shape (for the next call)."""
    msg: LLMMessage = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in tool_calls
        ]
    return msg


def _tool_message(name: str, content: str) -> LLMMessage:
    return {"role": "tool", "name": name, "content": content}


def _serialize_result(result: ToolResult) -> str:
    payload = result.data if result.ok else {"error": result.error}
    try:
        text = json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(payload)
    return text[:_MAX_TOOL_CONTENT]


def make_act_node(
    deps: GraphDeps,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    registry = deps.tool_registry
    gate = deps.tool_gate

    async def act(state: AgentState) -> dict[str, Any]:
        _emit("node", node="act")
        allowed = state.policy.allowed_tools
        tool_schemas = registry.to_ollama_schemas(allowed)

        response = await deps.llm.chat(
            state.messages, tools=tool_schemas or None, options=_ACT_OPTIONS
        )
        tokens = response.prompt_tokens + response.completion_tokens
        assistant_msg = _assistant_message(response.content, response.tool_calls)

        # No tools requested → this turn's content is the answer for the subgraph.
        if not response.tool_calls:
            if response.content:
                _emit("token", text=response.content, node="act")
            return {
                "messages": [assistant_msg],
                "final_answer": response.content or state.final_answer,
                "budget": state.budget.with_delta(tokens=tokens, iterations=1),
            }

        # Pass 1 — classify every requested call without running anything, and
        # collect the ones that need human approval.
        needs_approval = [
            tc
            for tc in response.tool_calls
            if tc["name"] in registry
            and (d := gate.check(tc["name"], state.policy)).allowed
            and d.requires_approval
        ]
        approved = True
        if needs_approval:
            decision = interrupt(
                {
                    "pending": [
                        PendingApproval(tool=tc["name"], arguments=tc["arguments"]).model_dump()
                        for tc in needs_approval
                    ]
                }
            )
            approved = _is_approved(decision)

        # Pass 2 — execute in the order the model asked, applying the classification.
        new_messages: list[LLMMessage] = [assistant_msg]
        records: list[ToolCallRecord] = []
        tool_calls_run = 0
        for tc in response.tool_calls:
            name = str(tc["name"])
            args = tc["arguments"] if isinstance(tc["arguments"], dict) else {}
            _emit("tool_called", name=name, arguments=args, node="act")

            if name not in registry:
                feedback = (
                    f"Tool '{name}' does not exist. Available tools: "
                    f"{sorted(registry.names())}. Pick a valid tool or answer directly."
                )
                new_messages.append(_tool_message(name, feedback))
                records.append(
                    ToolCallRecord(name=name, arguments=args, ok=False, error="unknown_tool")
                )
                _emit("tool_result", name=name, ok=False, error="unknown_tool", node="act")
                continue

            gate_decision = gate.check(name, state.policy)
            if not gate_decision.allowed:
                msg = f"Tool '{name}' is blocked by policy: {gate_decision.reason}"
                new_messages.append(_tool_message(name, msg))
                records.append(ToolCallRecord(name=name, arguments=args, ok=False, error="blocked"))
                _emit("tool_result", name=name, ok=False, error="blocked", node="act")
                continue

            if gate_decision.requires_approval and not approved:
                msg = f"User denied execution of '{name}'."
                new_messages.append(_tool_message(name, msg))
                records.append(ToolCallRecord(name=name, arguments=args, ok=False, error="denied"))
                _emit("tool_result", name=name, ok=False, error="denied", node="act")
                continue

            result = await registry.get(name).run(args)
            tool_calls_run += 1
            new_messages.append(_tool_message(name, _serialize_result(result)))
            records.append(
                ToolCallRecord(name=name, arguments=args, ok=result.ok, error=result.error)
            )
            _emit("tool_result", name=name, ok=result.ok, error=result.error, node="act")

        return {
            "messages": new_messages,
            "tools_called": records,
            "budget": state.budget.with_delta(
                tokens=tokens, tool_calls=tool_calls_run, iterations=1
            ),
        }

    return act


def _is_approved(decision: Any) -> bool:
    """Interpret the value a human resumed the graph with as approve/deny."""
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, dict):
        return bool(decision.get("approved", False))
    if isinstance(decision, str):
        return decision.strip().lower() in {"approve", "approved", "yes", "true"}
    return False
