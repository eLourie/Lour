# ADR-004: Skills vs Tools — strict separation

**Status:** Accepted
**Date:** 2026-07-09

## Context

The single most common design mistake in agent systems is conflating two very
different things: the *high-level scenarios* a user invokes ("research this
topic", "review this code") and the *atomic functions* the LLM calls
(`web_search`, `code_exec`). Collapsing them yields a system that is neither a
clean public API nor a clean tool interface — the demo-grade outcome
(PROJECT_CONTEXT §5).

## Decision

Model the two as **separate layers, each with its own registry and API**.

| | **Skill** | **Tool** |
|---|---|---|
| Abstraction | High-level business scenario | Low-level atomic operation |
| Invoked by | User (API) or router | LLM inside an agent |
| Lifetime | Multi-step, minutes | One call, seconds |
| State | Stateful (agent state) | Stateless |
| Defined in | `skills/definitions/*.yaml` (+ optional `.py`) | `tools/builtins/*.py` + `@tool` |
| Registry | `SkillRegistry` (YAML) | `ToolRegistry` (decorator) |
| Visibility | **Public** (`GET /v1/skills`) | Internal (`/v1/tools`, debug) |

Concretely, a **Skill** is a declaration:

```yaml
name: research_topic
agent: researcher            # skill declares the agent → no re-routing (§5.3)
tools_allowed: [web_search, web_fetch, rag_query]
input_schema: { topic: str, depth: quick|deep }
budget: { max_cost_tokens: 50000, max_duration_s: 300 }
```

- **`SkillRegistry`** discovers `definitions/*.yaml` and, when present, a matching
  `implementations/<name>.py` `Skill` subclass (the post-processing override hook,
  e.g. `review_code`). Adding a skill = drop a file; no registry edits.
- **`SkillRouter`** performs the single free-text→skill classification for
  `/v1/skills/auto` (§5.3); the skill's YAML then supplies the agent, so there is
  no second supervisor routing pass.
- A skill **drives the existing supervisor graph** with a *forced* route: it seeds
  `state.route` so the `route` node short-circuits its LLM classification and the
  declared agent owns the run.

## Consequences

- A **visible catalogue** of capabilities (`GET /v1/skills`) — important for the
  demo and for any UI to build invocation forms against.
- **Per-skill policy** (budget + tool allowlist) and **per-skill traceability**
  (each invocation opens a Session tagged with the skill name).
- **Declarative extensibility**: a new capability is a YAML file plus a test.
- Tools stay private building blocks the LLM composes; their surface never leaks
  into the public API.

## Trade-offs

- Two registries and two mental models instead of one.
- Mitigation: they share the same enforcement substrate — a skill's
  `tools_allowed` folds into `Policy.allowed_tools`, enforced by the same ToolGate
  the agents already use (ADR-011). The skill layer adds declaration and routing,
  not a parallel execution path.
