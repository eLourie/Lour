# ADR-007: Native function calling + graceful degradation

**Status:** Accepted
**Date:** 2026-07-09

## Context

The agent loop is a chain of LLM calls that decide which tools to invoke. There
are two ways to get tools out of a model: parse them from free text (ReAct-style)
or use the model's **native function-calling** channel. The reference model
(`qwen3:14b`) supports native tool calls but, as a 14B-class model, is only
*best-effort* on long multi-step chains (PROJECT_CONTEXT §3.2).

## Decision

Use **native Ollama function calling** in the `act` node (not ReAct string
parsing), and make the loop robust to a smaller model's mistakes rather than
assuming a bigger one.

Mitigations, all implemented in the orchestration layer:

- **Tool-name validation** — a call to a tool not in the registry becomes a
  corrective `tool` message ("that tool does not exist; pick a valid one"), and
  the model retries with feedback on the next iteration. It never crashes the run.
- **Retry-with-feedback** — the same mechanism surfaces tool failures (validation
  errors, blocked-by-policy) back into the transcript so the model can recover.
- **ToolGate enforcement** — the resolved policy's allowlist is applied before a
  tool runs; side-effecting tools flagged for approval trigger a single HITL
  interrupt *before any tool in the turn executes*, so a resumed run never
  double-executes (ADR-011).
- **Budget + loop detection** — the BudgetEnforcer forces finalisation when the
  iteration/tool-call/token/wall-time budget is spent or when the same tool call
  repeats without progress.
- **Deterministic tool calls** — the act LLM call runs at temperature 0, which
  both improves tool-calling reliability and keeps an interrupted-and-resumed
  iteration consistent.

Reliability-critical long chains can be pointed at a cloud provider through the
same `LLMProvider` Protocol (ADR-002) without touching the graph.

## Consequences

- Higher-quality tool calls and far fewer parsing errors than ReAct.
- A single bad tool call is a recoverable event, not a failure — the loop is a
  *capability tier*, not a fragile happy-path.

## Trade-offs

- Local tool-calling on 14B is best-effort; some long chains still need the cloud
  tier. This replaces the earlier, unrealisable "needs 32B+" requirement (32B does
  not fit in 24 GB, PROJECT_CONTEXT §3.3) with graceful degradation plus an
  optional cloud escape hatch.
