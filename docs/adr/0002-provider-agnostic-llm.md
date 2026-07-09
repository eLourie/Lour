# ADR-002: Provider-agnostic LLM layer (Ollama = dev/test tier)

**Status:** Accepted  
**Date:** 2026-07-03

## Context

The agent loop makes N sequential LLM calls. A 14B local model is reliable
for short chains (2–3 tool calls) but loses coherence on long multi-step
sequences. Cloud models are reliable but add latency and cost.

## Decision

Define `LLMProvider` as a `Protocol`. Ollama is the **local dev/test tier** —
it serves all calls by default. Cloud providers (Anthropic, OpenAI) are the
**reliability tier** — selectable via `LLM_PROVIDER` env var for long chains.

This is **not** a migration plan from Ollama to vLLM — it is a multi-provider
abstraction where both tiers coexist.

## Consequences

- Changing `LLM_PROVIDER=ollama→anthropic` requires no code changes.
- Local fully-offline scenario works end-to-end with Ollama.
- Cloud provider is optional; the system degrades gracefully without it.
- vLLM is a valid future addition (same Protocol, new factory branch).

## Trade-offs

- Local tool-calling on 14B is best-effort on long chains.
- Mitigations: retry-with-feedback, tool-name validation, budget enforcement (ADR-007).