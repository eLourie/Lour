# Architecture overview

Lour is a production-grade, single-user multi-agent system: a LangGraph
supervisor over local (Ollama) or cloud LLMs, hybrid RAG, a three-layer memory,
and an extensible tool system exposed over MCP. This document is the practical
map; the full design rationale lives in `PROJECT_CONTEXT_2.0.md` and the ADRs
under [`docs/adr/`](adr/).

Diagrams (render at [mermaid.live](https://mermaid.live) or with `mmdc`):
[`diagrams/system.mmd`](diagrams/system.mmd) ·
[`diagrams/agent_flow.mmd`](diagrams/agent_flow.mmd).

## Shape of the system

One deployable app (a **modular monolith**, ADR-010). Everything external —
Ollama, Postgres, Redis, Qdrant, the reranker, Langfuse, the code sandbox, and
any cloud LLM — is an **attached resource** addressed from `.env` (twelve-factor).
Changing "all on one Mac" ↔ "spread across the cloud" is a config change, not a
code change. There are three topology profiles (Solo / Split / Offloaded);
the default and reference is **Split** on an M4 24 GB with `qwen3:14b`.

## The six code layers

Requests flow top-to-bottom; dependencies only ever point downward.

| # | Layer | Responsibility | Key pieces |
|---|-------|----------------|------------|
| 1 | **Gateway** (`app/gateway`) | Everything before business logic | auth middleware (API-key core / JWT showcase), rate limiting, security headers, SSE, routes |
| 2 | **Skills** (`app/skills`) | Public catalogue of high-level capabilities | YAML declarations, `SkillRegistry`, `SkillRouter` |
| 3 | **Orchestration** (`app/agents`) | LangGraph supervisor + sub-agents | `AgentState`, nodes, subgraphs, `PostgresSaver`, `BudgetEnforcer`, HITL |
| 4 | **Tools** (`app/tools`) | Atomic LLM-callable functions | `ToolRegistry` + `@tool`, `ToolGate`, sandbox dispatch, MCP client/server |
| 5 | **Services** (`app/services`) | Transport-agnostic domain logic | LLM, Embeddings, Reranker, RAG, Memory, Sandbox, Ingestion |
| 6 | **Infrastructure** (`app/infra`) | Thin async clients + persistence | Ollama/Qdrant/Redis/Postgres clients, Repository + Unit of Work |

**Cross-cutting** (`app/core`) — config, structlog logging, telemetry, the
`AppError` hierarchy, security, and the unified `Policy` — threads through every
layer via DI, middleware and decorators; it is not itself a layer.

## Two invariants worth knowing

- **Skills vs Tools** (ADR-004). A *Skill* is a high-level scenario a user
  invokes (`research_topic`); a *Tool* is an atomic function the LLM calls
  (`web_search`). Separate registries, separate APIs. A skill declares its
  agent, so choosing a skill also chooses the agent — routing happens once.
- **Policy: declare once, enforce twice** (ADR-011). One schema
  `Policy{budget, allowed_tools, approval_rules}`, composed by `PolicyResolver`
  (defaults ← skill ← agent ← request, most-restrictive-wins), enforced at two
  natural points: the **BudgetEnforcer** in the graph (iterations / tokens /
  wall-time) and the **ToolGate** at the tool boundary (allowlist + HITL
  approval).

## Request flows

**Free-form chat** (`POST /v1/chat`, streamed as SSE). The supervisor graph runs
`memory_recall → route → {researcher | coder | direct} → memory_write → END`.
Routing is a structured Pydantic `Route` (not string parsing). The response
streams two levels at once (`app/agents/events.py`): coarse node/tool events for
a timeline, and token frames for the answer. Checkpoints persist to Postgres
after every node (ADR-008), so a killed session resumes from its last
checkpoint, and a tool that needs approval pauses via `interrupt` and resumes
through `POST /v1/sessions/{id}/approve`.

**Skill invocation** (`POST /v1/skills/{name}/invoke`, synchronous JSON). The
skill's YAML declares the agent, tool allowlist and per-skill policy; the run
enters that agent directly. `POST /v1/skills/auto` classifies free text to a
skill first (§5.3).

**RAG query** (`POST /v1/rag/query`). Hybrid retrieval: dense (bge-m3 via Ollama)
+ sparse (BM42 via FastEmbed) → RRF fusion → cross-encoder rerank (served
**outside** Ollama, lazy-loaded on MPS) → top-K with metadata filters.

## Provider tiers

`LLMProvider` is a Protocol (ADR-002). **Ollama is the local dev/test tier**;
cloud providers (Anthropic / OpenAI) are an optional **reliability tier** for
long multi-step tool chains where a local 14B loses coherence. Switching is
`LLM_PROVIDER=ollama|anthropic|openai` — no code changes. The fully local path
is a first-class supported scenario.

## Observability

`structlog` JSON logging with request/session/trace contextvars is the
always-on baseline (works offline). Langfuse Cloud layers on top via `@observe`
when `TELEMETRY_BACKEND=langfuse_cloud`. Layer latency, tool success rate,
retrieval recall and cloud cost are tracked in `app/core/metrics.py` and printed
by `make eval`.

## Where to go next

- Extending the system (tools, skills, providers, MCP, hardware):
  [`extending.md`](extending.md).
- Why each choice was made: the twelve ADRs in [`adr/`](adr/).
- Running it: the [README](../README.md) quickstart.
