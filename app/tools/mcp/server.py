"""
app/tools/mcp/server.py

MCP server (showcase, ADR-006): expose this project's builtin tools to any MCP
client — Claude Desktop, Cursor, etc. Bidirectional MCP: we consume external
servers (client.py) AND publish our own tools here.

Run standalone over stdio (how Claude Desktop launches it):

    uv run python -m app.tools.mcp.server

Only the dependency-light builtins are published here (no live PG/Qdrant
needed): datetime, web_search, web_fetch, http_request, code_exec. Each MCP
function delegates to the real BaseTool instance, so behaviour matches the API.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from app.core.config import get_settings
from app.services.sandbox.docker_sandbox import DockerSandbox
from app.tools.builtins.code_exec import CodeExecTool
from app.tools.builtins.datetime_tool import DateTimeTool
from app.tools.builtins.http_request import HttpRequestTool
from app.tools.builtins.web_fetch import WebFetchTool
from app.tools.builtins.web_search import WebSearchTool

if TYPE_CHECKING:
    from app.tools.base import ToolResult


def _dump(result: ToolResult) -> str:
    """Serialise a ToolResult for return over MCP (JSON text)."""
    return json.dumps(result.model_dump(), default=str, ensure_ascii=False)


def build_server() -> FastMCP:
    """Assemble the FastMCP server with builtin tools wired in."""
    settings = get_settings()
    mcp = FastMCP("lour-tools")

    datetime_tool = DateTimeTool()
    web_search_tool = WebSearchTool(settings)
    web_fetch_tool = WebFetchTool()
    http_tool = HttpRequestTool(settings.tools.http_allowlist)
    code_tool = CodeExecTool(DockerSandbox(settings.sandbox))

    @mcp.tool(description=datetime_tool.description)
    async def get_datetime(timezone: str = "UTC") -> str:
        return _dump(await datetime_tool.run({"timezone": timezone}))

    @mcp.tool(description=web_search_tool.description)
    async def web_search(query: str, max_results: int = 5) -> str:
        return _dump(await web_search_tool.run({"query": query, "max_results": max_results}))

    @mcp.tool(description=web_fetch_tool.description)
    async def web_fetch(url: str) -> str:
        return _dump(await web_fetch_tool.run({"url": url}))

    @mcp.tool(description=http_tool.description)
    async def http_request(
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> str:
        return _dump(
            await http_tool.run(
                {"url": url, "method": method, "headers": headers, "json_body": json_body}
            )
        )

    @mcp.tool(description=code_tool.description)
    async def code_exec(code: str, language: str = "python") -> str:
        return _dump(await code_tool.run({"code": code, "language": language}))

    return mcp


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
