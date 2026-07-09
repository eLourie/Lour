#!/usr/bin/env python
"""
docker/init/qdrant/memory_collection.py

Initialise the Qdrant ``memories`` collection used by long-term memory.

Like the documents collection, this is created idempotently at app startup
(see app/main.py lifespan). This standalone script provisions it out-of-band —
e.g. from CI or a fresh environment before the app boots.

Only the dense named vector is used by long-term memory; the collection still
declares a sparse vector for schema symmetry with ``documents``.

Usage:
    uv run python docker/init/qdrant/memory_collection.py
"""

from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.infra.clients.qdrant import QdrantClient


async def main() -> None:
    settings = get_settings()
    client = QdrantClient(settings.qdrant)
    try:
        await client.ensure_collection(settings.qdrant.collection_memory)
        print(f"✓ Qdrant collection ready: {settings.qdrant.collection_memory}")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
