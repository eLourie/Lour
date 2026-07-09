#!/usr/bin/env python
"""
docker/init/qdrant/bootstrap.py

Initialise the Qdrant ``documents`` collection with named vectors
(dense + sparse) for hybrid search.

The app also bootstraps collections idempotently at startup (see
app/main.py lifespan). This standalone script is for provisioning Qdrant
out-of-band — e.g. from CI, a seed job, or a fresh environment before the
app boots.

Usage:
    uv run python docker/init/qdrant/bootstrap.py
"""

from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.infra.clients.qdrant import QdrantClient


async def main() -> None:
    settings = get_settings()
    client = QdrantClient(settings.qdrant)
    try:
        await client.ensure_collection(settings.qdrant.collection_docs)
        print(f"✓ Qdrant collection ready: {settings.qdrant.collection_docs}")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
