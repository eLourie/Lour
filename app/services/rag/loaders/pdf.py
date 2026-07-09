"""
app/services/rag/loaders/pdf.py

PDF loader: extracts text and basic page metadata via pypdf.

pypdf is synchronous, so extraction runs in a worker thread.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.exceptions import ValidationError
from app.services.rag.loaders.base import LoadedDocument


class PdfLoader:
    doc_type = "pdf"

    def supports(self, source: str) -> bool:
        return source.lower().endswith(".pdf")

    async def load(self, source: str) -> LoadedDocument:
        path = Path(source)
        if not path.is_file():
            raise ValidationError(f"PDF not found: {source}")
        return await asyncio.to_thread(self._load_sync, path)

    def _load_sync(self, path: Path) -> LoadedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        content = "\n\n".join(pages).strip()

        title = getattr(reader.metadata, "title", None) or path.stem

        return LoadedDocument(
            content=content,
            source_uri=str(path),
            doc_type=self.doc_type,
            title=title,
            metadata={"page_count": len(pages), "filename": path.name},
        )
