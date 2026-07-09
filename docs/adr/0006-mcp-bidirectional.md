# ADR-006: Bidirectional MCP (client core + server showcase)

**Status:** Accepted  
**Date:** 2026-07-09

## Context

The Model Context Protocol (Anthropic) is the emerging industry standard for
wiring tools to LLMs. Two directions are possible: **consuming** external MCP
servers (filesystem, git, search, ...) and **exposing** our own tools to MCP
clients (Claude Desktop, Cursor). We already have a `BaseTool`/`ToolRegistry`
abstraction (Phase 3) — MCP should slot in as another tool *source/sink*, not a
parallel universe.

## Decision

Implement MCP in **both** directions behind the existing tool abstraction:

- **Client (core)** — `app/tools/mcp/client.py` connects to configured stdio
  servers, discovers their tools and invokes them. `app/tools/mcp/adapter.py`
  wraps each remote tool as a `BaseTool` (`mcp_<server>_<tool>`), so it flows
  through the same `ToolRegistry`, `ToolGate` and schema path as a builtin. The
  client is **dormant** when no servers are configured (`MCP_SERVERS_JSON`
  empty) and one server failing to start never aborts app startup.

- **Server (showcase)** — `app/tools/mcp/server.py` publishes the
  dependency-light builtins over `FastMCP`/stdio for any MCP client to call
  (`uv run python -m app.tools.mcp.server`).

The client is **core** (real leverage: reuse the whole MCP ecosystem); the
server is **showcase** (a strong demo of interoperability, not required for the
MVP path).

## Consequences

- Remote MCP tools are indistinguishable from builtins to the agent — same
  contract, same policy enforcement, same observability.
- Adding an external capability is config-only (`MCP_SERVERS_JSON`), no code.
- The server lets external editors drive our sandbox/search/fetch tools.

## Trade-offs

- Extra moving parts (stdio session lifecycle, an AsyncExitStack owning
  connections). Mitigation: the common `BaseTool` keeps MCP as a thin adapter,
  and the client fails soft (dormant / per-server error isolation).
- The showcase server exposes only builtins that need no live PG/Qdrant, to stay
  self-contained as a standalone process.
