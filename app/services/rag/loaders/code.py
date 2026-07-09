"""
app/services/rag/loaders/code.py

Code loader: parses source with tree-sitter and chunks by symbol boundaries
(top-level functions / classes) instead of arbitrary character windows, so a
retrieved chunk is a coherent unit of code.

The parsed segments are returned via ``LoadedDocument.segments`` — the
ingestion pipeline indexes them directly and skips the semantic chunker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.services.rag.loaders.base import LoadedDocument

logger = get_logger(__name__)

# File extension → tree-sitter-language-pack grammar name.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
}

# Node types that represent a top-level definition worth its own chunk.
_DEFINITION_SUFFIXES = ("_definition", "_declaration", "_item")
_DEFINITION_KEYWORDS = ("function", "class", "method", "struct", "impl", "interface")


class CodeLoader:
    doc_type = "code"

    def supports(self, source: str) -> bool:
        return Path(source).suffix.lower() in _EXT_TO_LANG

    async def load(self, source: str) -> LoadedDocument:
        path = Path(source)
        if not path.is_file():
            raise ValidationError(f"Source file not found: {source}")
        lang = _EXT_TO_LANG.get(path.suffix.lower())
        if lang is None:
            raise ValidationError(f"Unsupported code extension: {path.suffix}")
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        segments = await asyncio.to_thread(self._chunk_by_symbols, content, lang)

        return LoadedDocument(
            content=content,
            source_uri=str(path),
            doc_type=self.doc_type,
            title=path.name,
            metadata={"filename": path.name, "language": lang},
            segments=segments or None,
        )

    def _chunk_by_symbols(self, content: str, lang: str) -> list[str]:
        """Split source into top-level definitions; empty list on parse failure."""
        try:
            from tree_sitter_language_pack import get_parser

            parser = get_parser(lang)
        except Exception as exc:  # pragma: no cover - grammar load edge case
            logger.warning("code_parser_unavailable", language=lang, error=str(exc))
            return []

        data = content.encode("utf-8")
        tree = parser.parse(data)
        segments: list[str] = []

        for node in tree.root_node.children:
            if self._is_definition(node.type):
                text = data[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
                if text.strip():
                    segments.append(text)

        return segments

    @staticmethod
    def _is_definition(node_type: str) -> bool:
        if node_type.endswith(_DEFINITION_SUFFIXES):
            return True
        return any(kw in node_type for kw in _DEFINITION_KEYWORDS)
