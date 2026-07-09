"""
app/tools/mcp/client.py

MCP client (core, ADR-006): connect to external MCP servers over stdio, discover
their tools and invoke them. This lets the agent reuse the large ecosystem of
existing MCP servers (filesystem, git, ...) through our own ToolRegistry.

Lifecycle:
  - ``connect()`` opens one stdio session per configured server and lists tools.
  - ``call()`` invokes a remote tool and returns its textual content.
  - ``aclose()`` tears every session down (owned by an AsyncExitStack).

When no servers are configured the client stays dormant: ``connect()`` is a
no-op and ``discovered_tools()`` returns nothing.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)


class RemoteTool(BaseModel):
    """Metadata for a tool discovered on a remote MCP server."""

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]


class McpClient:
    """Manages stdio connections to external MCP servers."""

    def __init__(self, servers: Mapping[str, dict[str, Any]]) -> None:
        self._servers = dict(servers)
        self._stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}
        self._tools: list[RemoteTool] = []

    @property
    def connected(self) -> bool:
        return bool(self._sessions)

    async def connect(self) -> None:
        """Open a session to each configured server and list its tools."""
        if not self._servers:
            logger.debug("mcp_client_dormant", reason="no servers configured")
            return

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        for name, cfg in self._servers.items():
            try:
                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=cfg.get("env"),
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[name] = session

                listed = await session.list_tools()
                for t in listed.tools:
                    self._tools.append(
                        RemoteTool(
                            server=name,
                            name=t.name,
                            description=t.description or "",
                            input_schema=dict(t.inputSchema or {}),
                        )
                    )
                logger.info("mcp_server_connected", server=name, tools=len(listed.tools))
            except Exception as exc:
                logger.warning("mcp_server_connect_failed", server=name, error=str(exc))

    def discovered_tools(self) -> list[RemoteTool]:
        return list(self._tools)

    async def call(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke ``tool`` on ``server`` and return its collected content."""
        session = self._sessions.get(server)
        if session is None:
            raise ExternalServiceError(f"MCP server {server!r} is not connected")
        try:
            result = await session.call_tool(tool, arguments=arguments)
        except Exception as exc:
            raise ExternalServiceError(f"MCP call {server}/{tool} failed: {exc}") from exc
        return _collect_content(result)

    async def aclose(self) -> None:
        await self._stack.aclose()
        self._sessions.clear()


def _collect_content(result: Any) -> Any:
    """Flatten an MCP CallToolResult into plain text / structured data."""
    if getattr(result, "isError", False):
        text = _texts(result)
        raise ExternalServiceError(f"MCP tool returned an error: {text}")
    texts = _texts(result)
    if len(texts) == 1:
        return texts[0]
    return texts


def _texts(result: Any) -> list[str]:
    out: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            out.append(text)
    return out
