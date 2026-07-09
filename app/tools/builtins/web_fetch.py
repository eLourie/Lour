"""
app/tools/builtins/web_fetch.py

web_fetch — fetch a URL and return its readable main content (boilerplate
stripped via trafilatura). Mirrors the RAG UrlLoader extraction path.
"""

from __future__ import annotations

import asyncio

import httpx
from pydantic import BaseModel, Field

from app.core.exceptions import ExternalServiceError
from app.tools.base import BaseTool, ToolResult
from app.tools.decorators import audited, retried
from app.tools.registry import tool

_TIMEOUT_S = 20.0
_MAX_CHARS = 20_000
_USER_AGENT = "Mozilla/5.0 (compatible; LourAgent/1.0; +https://github.com/lour)"


class WebFetchArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL of the page to fetch.")


@tool
class WebFetchTool(BaseTool[WebFetchArgs]):
    name = "web_fetch"
    description = (
        "Fetch a web page and return its readable text content with boilerplate "
        "removed. Use to read a specific known URL. Do NOT use to search the web "
        "(use web_search) or to call JSON APIs (use http_request)."
    )
    args_schema = WebFetchArgs

    @audited
    @retried(attempts=3, exceptions=(ExternalServiceError,))
    async def execute(self, args: WebFetchArgs) -> ToolResult:
        if not args.url.startswith(("http://", "https://")):
            return ToolResult.failure("url must be an absolute http(s) URL")
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT_S,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                response = await client.get(args.url)
                response.raise_for_status()
                html = response.text
        except httpx.HTTPStatusError as exc:
            raise ExternalServiceError(
                f"Fetch failed with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise ExternalServiceError(f"Fetch failed: {exc}") from exc

        content, title = await asyncio.to_thread(self._extract, html, args.url)
        if not content.strip():
            return ToolResult.failure(f"No readable content extracted from {args.url}")
        truncated = content[:_MAX_CHARS]
        return ToolResult.success(
            {"url": args.url, "title": title, "content": truncated},
            truncated=len(content) > _MAX_CHARS,
        )

    @staticmethod
    def _extract(html: str, url: str) -> tuple[str, str | None]:
        import trafilatura

        content = trafilatura.extract(html, url=url, include_comments=False) or ""
        title: str | None = None
        meta = trafilatura.extract_metadata(html)
        if meta is not None and meta.title:
            title = meta.title
        return content, title
