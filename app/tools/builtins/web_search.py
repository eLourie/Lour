"""
app/tools/builtins/web_search.py

web_search — search the web via an env-switchable provider (Tavily or SearXNG),
both behind one tool interface (Open Question §9.3). The tool does not know which
backend answered; only the config decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from app.core.config import WebSearchProvider
from app.core.exceptions import ConfigurationError, ExternalServiceError
from app.tools.base import BaseTool, ToolResult
from app.tools.decorators import audited, retried
from app.tools.registry import tool

if TYPE_CHECKING:
    from app.core.config import Settings

_TIMEOUT_S = 20.0
_TAVILY_URL = "https://api.tavily.com/search"


class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query.")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of results to return.")


@tool
class WebSearchTool(BaseTool[WebSearchArgs]):
    name = "web_search"
    description = (
        "Search the web and return ranked results with title, url and snippet. "
        "Use when you need current information not in the knowledge base. "
        "Do NOT use to fetch a specific known URL (use web_fetch)."
    )
    args_schema = WebSearchArgs

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @audited
    @retried(attempts=2, exceptions=(ExternalServiceError,))
    async def execute(self, args: WebSearchArgs) -> ToolResult:
        provider = self._settings.web_search_provider
        if provider is WebSearchProvider.TAVILY:
            results = await self._tavily(args)
        else:
            results = await self._searxng(args)
        return ToolResult.success(results, provider=str(provider), count=len(results))

    async def _tavily(self, args: WebSearchArgs) -> list[dict[str, Any]]:
        api_key = self._settings.tavily_api_key
        if not api_key:
            raise ConfigurationError("TAVILY_API_KEY is not set")
        payload = {
            "api_key": api_key,
            "query": args.query,
            "max_results": args.max_results,
        }
        data = await self._post_json(_TAVILY_URL, payload)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "score": r.get("score"),
            }
            for r in data.get("results", [])
        ]

    async def _searxng(self, args: WebSearchArgs) -> list[dict[str, Any]]:
        base = self._settings.searxng_base_url.rstrip("/")
        params = {"q": args.query, "format": "json"}
        data = await self._get_json(f"{base}/search", params)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "score": r.get("score"),
            }
            for r in data.get("results", [])[: args.max_results]
        ]

    @staticmethod
    async def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                result: dict[str, Any] = resp.json()
                return result
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"web_search request failed: {exc}") from exc

    @staticmethod
    async def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                result: dict[str, Any] = resp.json()
                return result
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"web_search request failed: {exc}") from exc
