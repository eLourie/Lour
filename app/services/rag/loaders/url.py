"""
app/services/rag/loaders/url.py

URL loader: fetches a page and extracts the readable main content via
trafilatura (boilerplate / nav / ads stripped).
"""

from __future__ import annotations

import asyncio

import httpx

from app.core.exceptions import ExternalServiceError, ValidationError
from app.core.logging import get_logger
from app.services.rag.loaders.base import LoadedDocument

logger = get_logger(__name__)

_TIMEOUT_S = 20.0
_USER_AGENT = "Mozilla/5.0 (compatible; LourRAG/1.0; +https://github.com/lour)"


class UrlLoader:
    doc_type = "url"

    def supports(self, source: str) -> bool:
        return source.startswith(("http://", "https://"))

    async def load(self, source: str) -> LoadedDocument:
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT_S,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                response = await client.get(source)
                response.raise_for_status()
                html = response.text
        except httpx.HTTPStatusError as exc:
            raise ExternalServiceError(
                f"URL fetch failed with HTTP {exc.response.status_code}: {source}"
            ) from exc
        except httpx.RequestError as exc:
            raise ExternalServiceError(f"URL fetch failed: {source} ({exc})") from exc

        content, title = await asyncio.to_thread(self._extract, html, source)
        if not content.strip():
            raise ValidationError(f"No readable content extracted from URL: {source}")

        return LoadedDocument(
            content=content,
            source_uri=source,
            doc_type=self.doc_type,
            title=title,
            metadata={"url": source},
        )

    @staticmethod
    def _extract(html: str, source: str) -> tuple[str, str | None]:
        import trafilatura

        content = trafilatura.extract(html, url=source, include_comments=False) or ""
        title: str | None = None
        meta = trafilatura.extract_metadata(html)
        if meta is not None and meta.title:
            title = meta.title
        return content, title
