"""Loader registry helpers."""

from __future__ import annotations

from app.services.rag.loaders.base import LoadedDocument, Loader
from app.services.rag.loaders.code import CodeLoader
from app.services.rag.loaders.markdown import MarkdownLoader
from app.services.rag.loaders.pdf import PdfLoader
from app.services.rag.loaders.url import UrlLoader


def default_loaders() -> list[Loader]:
    """Return the built-in loaders in resolution order."""
    return [UrlLoader(), PdfLoader(), MarkdownLoader(), CodeLoader()]


__all__ = [
    "CodeLoader",
    "LoadedDocument",
    "Loader",
    "MarkdownLoader",
    "PdfLoader",
    "UrlLoader",
    "default_loaders",
]
