"""
app/services/rag/loaders/markdown.py

Markdown loader: keeps the raw markdown as content and extracts the heading
outline into metadata so retrieval can surface section context.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from app.core.exceptions import ValidationError
from app.services.rag.loaders.base import LoadedDocument

# ATX headings: leading #'s, ignoring fenced code blocks handled separately.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$", re.MULTILINE)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


class MarkdownLoader:
    doc_type = "markdown"

    def supports(self, source: str) -> bool:
        return source.lower().endswith((".md", ".markdown", ".mdx"))

    async def load(self, source: str) -> LoadedDocument:
        path = Path(source)
        if not path.is_file():
            raise ValidationError(f"Markdown file not found: {source}")
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        headings = self._extract_headings(content)
        title = headings[0][1] if headings else path.stem

        return LoadedDocument(
            content=content,
            source_uri=str(path),
            doc_type=self.doc_type,
            title=title,
            metadata={
                "filename": path.name,
                "headings": [{"level": lvl, "text": txt} for lvl, txt in headings],
            },
        )

    @staticmethod
    def _extract_headings(text: str) -> list[tuple[int, str]]:
        """Return (level, text) for each heading, ignoring fenced code blocks."""
        stripped = _FENCE_RE.sub("", text)
        return [(len(m.group(1)), m.group(2).strip()) for m in _HEADING_RE.finditer(stripped)]
