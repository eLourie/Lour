# ADR-011: Unified Policy — one declaration, two enforcement points

**Status:** Accepted
**Date:** 2026-07-09

## Context

An early design duplicated policy across three places — skills, agents and tools
each carried their own `tools_allowed` and HITL rules. There was no single source
of truth and no clear answer to "who wins when they disagree". With the skills
layer (ADR-004) now declaring budgets and tool allowlists too, this duplication
would have multiplied.

## Decision

**One schema, declared once, enforced in two natural places.**

```
Policy { budget, allowed_tools, approval_rules, requires_confirmation }
```

- **Declaration** lives with the data that owns it: system defaults (config),
  the skill (YAML), the agent, and the request. Nobody writes enforcement logic.
- **`PolicyResolver`** composes the effective policy across those layers with
  **most-restrictive-wins** semantics:
  `defaults ← skill ← agent ← request`. Budgets take the minimum; `allowed_tools`
  take the intersection (`None` = unbounded); approval/confirmation flags OR
  together. The resolved policy is stored on `AgentState` once at graph entry, so
  it survives a checkpoint resume.
- **Enforcement** happens at exactly two boundaries:
  - **`BudgetEnforcer`** (Orchestration layer) — iterations / tool calls / tokens
    / wall-time, plus loop detection.
  - **`ToolGate`** (Tools layer) — allowlist + HITL approval, checked at the
    tool-call boundary in the `act` node.

A skill contributes its layer for free: `SkillSpec.effective_policy()` folds
`tools_allowed` into `Policy.allowed_tools`, and its `budget` block into the
budget — so the same BudgetEnforcer and ToolGate that guard `/v1/chat` also guard
every skill invocation, with no skill-specific enforcement code.

## Consequences

- A single, inspectable notion of "what is this run allowed to do".
- No duplicated `tools_allowed` / approval rules across layers, and a defined
  conflict rule (most-restrictive-wins).
- Enforcement is centralised: two points, not scattered checks. Adding the skills
  layer required **zero** new enforcement code — only a new *declaration* source.

## Trade-offs

- Contributors must remember that policy is data, enforced elsewhere — you cannot
  "just add a check" in a skill.
- Mitigation: the two enforcement points are few and well-named, and
  `PolicyResolver` is the one place composition happens.
