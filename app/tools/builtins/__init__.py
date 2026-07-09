"""
app/tools/builtins/__init__.py

Builtin tool catalogue + factory.

``build_builtin_tools`` mirrors the RAG ``default_loaders()`` pattern: it
constructs every builtin tool with its runtime dependencies wired in and returns
them in a stable order. The lifespan registers the result into a ToolRegistry.

Importing this module also triggers the ``@tool`` decorators, populating the
discovery list used by the MCP server and introspection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.tools.builtins.code_exec import CodeExecTool
from app.tools.builtins.datetime_tool import DateTimeTool
from app.tools.builtins.filesystem import FileSystemTool
from app.tools.builtins.http_request import HttpRequestTool
from app.tools.builtins.rag_query import RagQueryTool
from app.tools.builtins.web_fetch import WebFetchTool
from app.tools.builtins.web_search import WebSearchTool

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.services.rag.retrieval import HybridRetriever
    from app.services.sandbox.base import SandboxService
    from app.tools.base import BaseTool


def build_builtin_tools(
    *,
    settings: Settings,
    retriever: HybridRetriever,
    sandbox: SandboxService,
) -> list[BaseTool[Any]]:
    """Instantiate all builtin tools with their dependencies (stable order)."""
    return [
        DateTimeTool(),
        WebSearchTool(settings),
        WebFetchTool(),
        RagQueryTool(retriever),
        FileSystemTool(settings.tools.workspace_dir),
        HttpRequestTool(settings.tools.http_allowlist),
        CodeExecTool(sandbox),
    ]


__all__ = [
    "CodeExecTool",
    "DateTimeTool",
    "FileSystemTool",
    "HttpRequestTool",
    "RagQueryTool",
    "WebFetchTool",
    "WebSearchTool",
    "build_builtin_tools",
]
