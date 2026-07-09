"""
tests/unit/test_loaders.py

Unit tests for loaders that only touch the local filesystem (markdown, code)
plus loader-selection logic. Network / PDF loaders are covered by integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.rag.loaders import default_loaders
from app.services.rag.loaders.code import CodeLoader
from app.services.rag.loaders.markdown import MarkdownLoader

pytestmark = pytest.mark.unit


def test_default_loaders_select_by_source() -> None:
    loaders = default_loaders()

    def resolve(source: str) -> str:
        return next(loader.doc_type for loader in loaders if loader.supports(source))

    assert resolve("https://example.com/page") == "url"
    assert resolve("/tmp/report.pdf") == "pdf"
    assert resolve("/tmp/notes.md") == "markdown"
    assert resolve("/tmp/main.py") == "code"


async def test_markdown_extracts_headings(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nIntro text.\n\n## Section A\n\nBody.\n", encoding="utf-8")

    doc = await MarkdownLoader().load(str(md))

    assert doc.doc_type == "markdown"
    assert doc.title == "Title"
    headings = doc.metadata["headings"]
    assert {h["text"] for h in headings} == {"Title", "Section A"}


async def test_markdown_ignores_headings_in_code_fences(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Real\n\n```\n# not a heading\n```\n", encoding="utf-8")

    doc = await MarkdownLoader().load(str(md))

    texts = [h["text"] for h in doc.metadata["headings"]]
    assert texts == ["Real"]


async def test_code_loader_chunks_by_symbol(tmp_path: Path) -> None:
    src = tmp_path / "mod.py"
    src.write_text(
        "import os\n\n"
        "def alpha():\n    return 1\n\n"
        "class Beta:\n    def m(self):\n        return 2\n",
        encoding="utf-8",
    )

    doc = await CodeLoader().load(str(src))

    assert doc.doc_type == "code"
    assert doc.metadata["language"] == "python"
    assert doc.segments is not None
    assert len(doc.segments) == 2  # def alpha + class Beta (import is not a definition)
    assert any("def alpha" in s for s in doc.segments)
    assert any("class Beta" in s for s in doc.segments)
