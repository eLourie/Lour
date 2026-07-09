"""
app/tools/mcp/adapter.py

Adapter: expose remote MCP tools through the local ToolRegistry so the agent
treats them exactly like builtins (same BaseTool contract, same ToolGate, same
schema path). ADR-006 — one BaseTool, MCP is just another source.

Naming: remote tools are namespaced ``mcp_<server>_<tool>`` so they never clash
with builtins and their origin stays visible in traces.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from app.tools.base import BaseTool, ToolResult
from app.tools.decorators import audited

if TYPE_CHECKING:
    from app.tools.mcp.client import McpClient, RemoteTool

_NAME_RE = re.compile(r"[^a-z0-9_]+")


class _McpPassthroughArgs(BaseModel):
    """Accepts any arguments — the remote server owns validation."""

    model_config = ConfigDict(extra="allow")


def _sanitize(raw: str) -> str:
    return _NAME_RE.sub("_", raw.lower()).strip("_")[:30]


class McpToolAdapter(BaseTool[_McpPassthroughArgs]):
    """Wraps one remote MCP tool as a local BaseTool."""

    args_schema = _McpPassthroughArgs

    def __init__(self, remote: RemoteTool, client: McpClient) -> None:
        self.name = _sanitize(f"mcp_{remote.server}_{remote.name}")
        self.description = (remote.description or f"Remote MCP tool {remote.name}")[:1024]
        self._client = client
        self._server = remote.server
        self._remote_name = remote.name
        self._input_schema = remote.input_schema

    def ollama_parameters(self) -> dict[str, Any]:
        # Surface the remote server's own JSON Schema to the model.
        return self._input_schema or {"type": "object", "properties": {}}

    @audited
    async def execute(self, args: _McpPassthroughArgs) -> ToolResult:
        payload = args.model_dump()
        data = await self._client.call(self._server, self._remote_name, payload)
        return ToolResult.success(data, server=self._server, remote_tool=self._remote_name)


def adapt_mcp_tools(client: McpClient) -> list[McpToolAdapter]:
    """Build a BaseTool adapter for every tool the client discovered."""
    return [McpToolAdapter(remote, client) for remote in client.discovered_tools()]
