# ADR-010: Deployment Topology — Modular Monolith + Twelve-Factor

**Status:** Accepted  
**Date:** 2026-07-03

## Context

The system is developed on an M4 MacBook (24 GB unified memory) but must be configurable for other hardware (16 GB Mac, NVIDIA workstation, cloud VM) and deployment topologies (everything local, split between local and cloud, fully offloaded) without code changes.

Additionally: the portfolio signal should demonstrate *disciplined* architecture, not over-engineering. Breaking the application into microservices for a single-user system would be a negative signal to a senior reviewer.

## Decision

**Modular monolith** (single deployable app with clear module boundaries = the six layers) + **twelve-factor backing services** (all external dependencies addressed via env vars, never hardcoded).

Three deployment profiles, switched via `DEPLOY_PROFILE` in `.env`:

| Profile | Local | External | Max model |
|---------|-------|----------|-----------|
| `solo` | everything incl. Langfuse self-host | nothing | ~7–8B |
| `split` (default) | Ollama + app + PG/Redis/Qdrant | Langfuse Cloud + optional cloud LLM | **14B** |
| `offloaded` | only Ollama + app | all backing services | 14B |

Backing services: Ollama, PostgreSQL, Redis, Qdrant, Reranker, Langfuse, cloud LLM APIs.  
Each is an *attached resource* — address from `BACKING_SERVICE_URL` in env, not in code.

## Why NOT microservices

Decomposing the app (e.g. separate "RAG service", "Memory service", "Tool service") for a single-user system would:
- Add inter-service latency with no throughput benefit (personal scale)
- Complicate local development significantly
- Be a textbook example of premature distributed systems complexity
- Send the wrong signal: "copied enterprise patterns without understanding tradeoffs"

Clear **module boundaries** (the six layers with typed interfaces and no cross-layer imports) demonstrates the same architectural discipline without the operational overhead.

## Consequences

- Code change to swap topology = zero; `.env` change = enough
- Ollama and reranker run on the host (not in Docker on Mac) — Metal not available in containers
- The `docker-compose.yml` manages only PG + Redis + Qdrant in the default `mac` profile
- Any future horizontal scaling path = split the modular monolith at module boundaries → still clean
