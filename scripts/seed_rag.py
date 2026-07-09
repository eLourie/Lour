#!/usr/bin/env python
"""
scripts/seed_rag.py

Seed the RAG corpus by ingesting a directory of files (and optionally a list of
URLs) through the running instance's ``/v1/rag/ingest`` API.

Bundled sample corpus lives in ``scripts/data/rag_corpus`` — a small starter
set across formats. Point ``--dir`` at your own folder (50-100 docs) to build
a real corpus for the Phase 2 DoD.

Usage:
    uv run python scripts/seed_rag.py
    uv run python scripts/seed_rag.py --dir ~/my-notes --urls-file urls.txt
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx

# Extensions handled by a file/source loader vs. read-and-send-as-text.
_SOURCE_EXTS = {".pdf", ".md", ".markdown", ".mdx", ".py", ".js", ".ts", ".tsx", ".go", ".rs",
                ".java", ".rb", ".c", ".h", ".cpp", ".cc", ".hpp"}
_TEXT_EXTS = {".txt", ".text", ".rst"}

_DEFAULT_DIR = Path(__file__).parent / "data" / "rag_corpus"


async def _ingest_one(client: httpx.AsyncClient, payload: dict[str, object]) -> str:
    resp = await client.post("/v1/rag/ingest", json=payload)
    resp.raise_for_status()
    data = resp.json()
    if data["skipped"]:
        return f"skip ({data['reason']})"
    return f"ok ({data['chunks']} chunks)"


async def seed(base_url: str, directory: Path, urls: list[str], force: bool) -> None:
    ok = skipped = failed = 0
    async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as client:
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in _SOURCE_EXTS:
                payload: dict[str, object] = {"source": str(path.resolve()), "force": force}
            elif ext in _TEXT_EXTS:
                payload = {
                    "text": path.read_text(encoding="utf-8", errors="replace"),
                    "title": path.name,
                    "doc_type": "text",
                    "force": force,
                }
            else:
                continue

            try:
                status = await _ingest_one(client, payload)
                print(f"  {path.name:<40} {status}")
                if status.startswith("skip"):
                    skipped += 1
                else:
                    ok += 1
            except Exception as exc:
                print(f"  {path.name:<40} FAILED: {exc}")
                failed += 1

        for url in urls:
            try:
                status = await _ingest_one(client, {"source": url, "force": force})
                print(f"  {url:<40} {status}")
                ok += 1 if not status.startswith("skip") else 0
                skipped += 1 if status.startswith("skip") else 0
            except Exception as exc:
                print(f"  {url:<40} FAILED: {exc}")
                failed += 1

    print(f"\nSeed complete: {ok} ingested, {skipped} skipped, {failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the RAG corpus via the API.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dir", type=Path, default=_DEFAULT_DIR)
    parser.add_argument("--urls-file", type=Path, default=None, help="One URL per line.")
    parser.add_argument("--force", action="store_true", help="Re-ingest unchanged content.")
    args = parser.parse_args()

    urls: list[str] = []
    if args.urls_file and args.urls_file.is_file():
        urls = [ln.strip() for ln in args.urls_file.read_text().splitlines() if ln.strip()]

    if not args.dir.is_dir():
        raise SystemExit(f"Corpus directory not found: {args.dir}")

    print(f"Seeding from {args.dir} ({len(urls)} URLs) → {args.base_url}\n")
    asyncio.run(seed(args.base_url, args.dir, urls, args.force))


if __name__ == "__main__":
    main()
