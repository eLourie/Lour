"""
app/infra/clients/qdrant.py

Async Qdrant client with collection-bootstrap helpers.
Named vectors are used so dense + sparse can live in one collection.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

from app.core.config import QdrantSettings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Named-vector keys used consistently across the codebase
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Dimensionality of bge-m3 dense embeddings
BGE_M3_DIM = 1024


class QdrantClient:
    """
    Wrapper around the official AsyncQdrantClient.

    Exposes the raw client via :attr:`client` and adds
    opinionated collection-bootstrap helpers so the rest of the
    codebase never hard-codes schema details.
    """

    def __init__(self, settings: QdrantSettings) -> None:
        self._settings = settings
        self.client = AsyncQdrantClient(
            host=settings.host,
            port=settings.port,
            prefer_grpc=False,
        )

    async def ensure_collection(
        self,
        name: str,
        *,
        dense_dim: int = BGE_M3_DIM,
    ) -> None:
        """
        Idempotently create a hybrid-search collection.

        The collection uses named vectors so dense and sparse can coexist.
        Safe to call at every startup — does nothing if already exists.
        """
        exists = await self.client.collection_exists(name)
        if exists:
            logger.debug("qdrant_collection_exists", collection=name)
            return

        await self.client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR: VectorParams(
                    size=dense_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info("qdrant_collection_created", collection=name)

    async def ping(self) -> bool:
        """Return True if Qdrant is reachable."""
        try:
            await self.client.get_collections()
            return True
        except Exception:
            logger.exception("Qdrant ping failed")
            return False

    async def aclose(self) -> None:
        await self.client.close()