# ADR-001: Technology Stack

**Status:** Accepted  
**Date:** 2026-07-03

## Context

We need a technology stack for a production-grade multi-agent AI system that:
- Runs locally on an M4 MacBook (24 GB unified memory) with Metal acceleration
- Is fully portable to other hardware/cloud via config (not code) changes
- Demonstrates senior-level engineering judgement for portfolio purposes
- Supports the complete agent capability surface: orchestration, RAG, memory, tools, MCP

## Decision

**Python 3.12** + **FastAPI** + **LangGraph** + **Ollama** + **Qdrant** + **PostgreSQL 16** + **Redis 7** + **Langfuse Cloud** + **uv** + **ruff** + **mypy strict**.

| Component | Choice | Key reason |
|-----------|--------|------------|
| Language | Python 3.12 | ML/LLM ecosystem standard, async, type hints |
| Package manager | uv | Speed, modern lockfile, pyproject.toml-native |
| Web framework | FastAPI | async-first, OpenAPI, mature |
| Orchestration | LangGraph | State machines, checkpointing, HITL, streaming |
| Local LLM runtime | Ollama | Best DX on Mac, native Metal, no Docker overhead |
| Vector store | Qdrant | Hybrid search (dense + BM42) out of the box, async client |
| RDBMS | PostgreSQL 16 | LangGraph checkpointer, metadata, episodic memory |
| Cache | Redis 7 | Short-term memory sliding window, rate limiting |
| Observability | Langfuse Cloud | LLM-native tracing, free tier, no memory overhead |
| Lint/format | ruff | Single tool replacing black + isort + flake8 |
| Type checker | mypy strict | Portfolio standard: zero type errors |

## Alternatives Considered

- **LlamaIndex Workflows** — less mature state machine support, weaker checkpointing
- **CrewAI / AutoGen** — opaque orchestration, harder to demonstrate understanding
- **pgvector** — weaker hybrid search; Qdrant has native BM42 sparse support
- **Weaviate** — heavier, less ergonomic Python client
- **Celery** for background jobs — overkill for single-user; APScheduler is enough (see ADR-012)
- **Poetry** — slower than uv; pyproject.toml support came later

## Consequences

- Ollama must run natively on the host (not in Docker on Mac — Metal not available in containers)
- Langfuse Cloud requires internet for LLM-native observability; structlog provides the offline baseline
- The stack is well-established enough that the portfolio signal comes from *how* it's composed, not which libraries are used
