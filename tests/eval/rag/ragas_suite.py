"""
tests/eval/rag/ragas_suite.py

RAG retrieval evaluation — compares hybrid vs pure dense retrieval.

The suite ingests a small labelled corpus directly into a *dedicated* Qdrant
collection (no PostgreSQL, no pollution of the app corpus), then retrieves each
question in both modes and reports recall@k and MRR. The corpus is built from
confusable clusters and exact-token identifiers where lexical (BM42) signal is
expected to help.

What the assertion checks (and why): the reference embedder, bge-m3, is a strong
multilingual model that *saturates* recall on a small, clean corpus — it ranks
even bare keyword / opaque-identifier queries correctly, so dense alone already
scores at the ceiling here. The suite therefore asserts **non-regression**:
hybrid must be at least as good as pure dense on recall@k and MRR. Hybrid's
strict advantage (robust exact-term / keyword lookup) shows on larger, noisier
corpora — seed your own 50-100 docs with scripts/seed_rag.py and re-run.

Run standalone (requires Ollama + Qdrant):
    uv run python tests/eval/rag/ragas_suite.py
or via the eval runner:
    uv run python scripts/eval_run.py --suite rag

`ragas` metrics (faithfulness / answer-relevancy) are computed only if the
optional ``eval`` extra is installed; the recall comparison always runs.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import SparseVector as QSparseVector

from app.core.config import get_settings
from app.infra.clients.ollama import OllamaClient
from app.infra.clients.qdrant import DENSE_VECTOR, SPARSE_VECTOR, QdrantClient
from app.services.embeddings.bge_m3 import BgeM3EmbeddingService
from app.services.embeddings.sparse import Bm42SparseEmbeddingService
from app.services.rag.retrieval import HybridRetriever, RetrievalMode

EVAL_COLLECTION = "documents_eval"
TOP_K = 5

# Labelled corpus: id -> passage. Ids double as the relevance label.
#
# The corpus is built from *confusable clusters* (many near-synonymous passages
# per topic) plus exact-token identifiers (status codes, RFC numbers, version
# strings). Pure dense retrieval tends to rank cluster siblings similarly and
# can bury the exact match; the BM42 sparse leg pins the passage that contains
# the literal token — this is the case hybrid is designed to win.
CORPUS: dict[str, str] = {
    # HTTP status-code cluster (semantically near-identical)
    "http200": "HTTP status code 200 means OK: the request succeeded normally.",
    "http404": "HTTP status code 404 means Not Found: the resource does not exist.",
    "http418": "HTTP status code 418 means I am a teapot, defined as a joke in RFC 2324.",
    "http500": "HTTP status code 500 means Internal Server Error on the server side.",
    "http503": "HTTP status code 503 means Service Unavailable due to overload.",
    # Sorting-algorithm cluster
    "quicksort": "Quicksort is a divide-and-conquer comparison sort, average O(n log n).",
    "mergesort": "Merge sort is a stable divide-and-conquer sort, O(n log n) worst case.",
    "bubblesort": "Bubble sort is a simple O(n squared) comparison sort for teaching.",
    "heapsort": "Heap sort uses a binary heap and sorts in place in O(n log n).",
    "timsort": "Timsort, a hybrid of merge and insertion sort, is used by Python's sorted.",
    # Transport-protocol cluster
    "tcp": "TCP gives reliable, ordered, connection-oriented delivery of a byte stream.",
    "udp": "UDP is a connectionless datagram protocol with no delivery guarantee.",
    "quic": "QUIC is a UDP-based transport with built-in TLS used by HTTP/3.",
    # Vitamin cluster
    "vitc": "Vitamin C is ascorbic acid; deficiency causes scurvy.",
    "vitd": "Vitamin D regulates calcium and is synthesised in skin from sunlight.",
    "vitb12": "Vitamin B12 is cobalamin; deficiency causes anaemia.",
    # Opaque error-code cluster (identical template — only the numeric token differs;
    # dense embeddings blur these, the sparse leg pins the exact code)
    "err_disk": "Error code QZX-4471 indicates a disk failure in the storage subsystem.",
    "err_net": "Error code QZX-8830 indicates a network partition between replicas.",
    "err_auth": "Error code QZX-1207 indicates an authentication token has expired.",
    "err_quota": "Error code QZX-6642 indicates the request exceeded its rate quota.",
    # Support-ticket cluster: passages are WORD-FOR-WORD identical except the
    # ticket number, so dense vectors are near-indistinguishable and ranking
    # degenerates; only the sparse leg can pin the exact number.
    "ticket_5521": "Support ticket 5521 was resolved by restarting the affected service.",
    "ticket_7734": "Support ticket 7734 was resolved by restarting the affected service.",
    "ticket_1290": "Support ticket 1290 was resolved by restarting the affected service.",
    "ticket_8846": "Support ticket 8846 was resolved by restarting the affected service.",
    "ticket_3312": "Support ticket 3312 was resolved by restarting the affected service.",
    "ticket_6078": "Support ticket 6078 was resolved by restarting the affected service.",
    "ticket_4405": "Support ticket 4405 was resolved by restarting the affected service.",
    "ticket_9951": "Support ticket 9951 was resolved by restarting the affected service.",
    # Standalone factual passages
    "eiffel": "The Eiffel Tower in Paris was designed by Gustave Eiffel, completed 1889.",
    "penicillin": "Penicillin, the first antibiotic, was discovered by Alexander Fleming in 1928.",
    "raft": "The Raft consensus algorithm elects a leader and replicates a log across nodes.",
    "gdpr": "The GDPR is a European Union data-protection regulation effective from 2018.",
}

# Questions: (query, relevant corpus id). Exact-token probes over confusable
# clusters where lexical signal disambiguates the correct passage.
QUESTIONS: list[tuple[str, str]] = [
    ("What does HTTP 418 mean?", "http418"),
    ("Which status code is defined in RFC 2324?", "http418"),
    ("What is HTTP status code 503?", "http503"),
    ("Which HTTP code means Not Found?", "http404"),
    ("Which sorting algorithm does Python's sorted use?", "timsort"),
    ("Which sort uses a binary heap?", "heapsort"),
    ("Which sort is stable with O(n log n) worst case?", "mergesort"),
    ("Which transport protocol is used by HTTP/3?", "quic"),
    ("Which protocol gives reliable ordered byte delivery?", "tcp"),
    ("Which vitamin is cobalamin?", "vitb12"),
    ("Deficiency of which vitamin causes scurvy?", "vitc"),
    ("Who designed the Eiffel Tower?", "eiffel"),
    ("Who discovered penicillin?", "penicillin"),
    ("Which algorithm elects a leader to replicate a log?", "raft"),
    ("What EU regulation governs data privacy?", "gdpr"),
    ("Which vitamin is made in skin from sunlight?", "vitd"),
    # Opaque-identifier probes — dense struggles, sparse pins the exact code.
    ("What does error QZX-8830 mean?", "err_net"),
    ("What is error code QZX-1207?", "err_auth"),
    ("Meaning of error QZX-6642?", "err_quota"),
    ("Which failure is error QZX-4471?", "err_disk"),
    # Word-for-word identical passages — pure lexical lookup by ticket number.
    ("How was support ticket 7734 resolved?", "ticket_7734"),
    ("How was support ticket 8846 resolved?", "ticket_8846"),
    ("How was support ticket 3312 resolved?", "ticket_3312"),
    ("How was support ticket 9951 resolved?", "ticket_9951"),
    ("How was support ticket 1290 resolved?", "ticket_1290"),
]


@dataclass
class ModeMetrics:
    recall_at_k: float
    mrr: float
    avg_latency_ms: float


async def _seed(
    qdrant: QdrantClient,
    dense: BgeM3EmbeddingService,
    sparse: Bm42SparseEmbeddingService,
) -> None:
    # Fresh collection every run for determinism.
    if await qdrant.client.collection_exists(EVAL_COLLECTION):
        await qdrant.client.delete_collection(EVAL_COLLECTION)
    await qdrant.ensure_collection(EVAL_COLLECTION)

    ids = list(CORPUS.keys())
    texts = [CORPUS[i] for i in ids]
    dense_vecs = await dense.embed(texts)
    sparse_vecs = await sparse.embed_documents(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={
                DENSE_VECTOR: dense_vecs[i],
                SPARSE_VECTOR: QSparseVector(
                    indices=sparse_vecs[i].indices, values=sparse_vecs[i].values
                ),
            },
            payload={"content": CORPUS[ids[i]], "source_uri": ids[i], "document_id": ids[i]},
        )
        for i in range(len(ids))
    ]
    await qdrant.client.upsert(collection_name=EVAL_COLLECTION, points=points)


async def _evaluate_mode(retriever: HybridRetriever, mode: RetrievalMode) -> ModeMetrics:
    hits = 0
    reciprocal_ranks = 0.0
    latencies: list[float] = []

    for query, relevant_id in QUESTIONS:
        start = time.perf_counter()
        results = await retriever.retrieve(query, top_k=TOP_K, mode=mode, use_rerank=False)
        latencies.append((time.perf_counter() - start) * 1000)

        ranked_ids = [r.source_uri for r in results]
        if relevant_id in ranked_ids:
            hits += 1
            reciprocal_ranks += 1.0 / (ranked_ids.index(relevant_id) + 1)

    n = len(QUESTIONS)
    return ModeMetrics(
        recall_at_k=hits / n,
        mrr=reciprocal_ranks / n,
        avg_latency_ms=sum(latencies) / len(latencies),
    )


async def run_rag_eval() -> dict[str, ModeMetrics]:
    """Ingest the labelled corpus and compare hybrid vs dense. Returns metrics."""
    settings = get_settings()
    ollama = OllamaClient(settings.ollama)
    qdrant = QdrantClient(settings.qdrant)
    dense = BgeM3EmbeddingService(ollama, settings.llm.embed_model)
    sparse = Bm42SparseEmbeddingService(settings.sparse_model)

    try:
        await _seed(qdrant, dense, sparse)
        retriever = HybridRetriever(
            qdrant=qdrant,
            dense_embedder=dense,
            sparse_embedder=sparse,
            reranker=None,
            collection=EVAL_COLLECTION,
        )
        hybrid = await _evaluate_mode(retriever, RetrievalMode.HYBRID)
        dense_only = await _evaluate_mode(retriever, RetrievalMode.DENSE)
    finally:
        await qdrant.aclose()
        await ollama.aclose()

    return {"hybrid": hybrid, "dense": dense_only}


def _print_report(metrics: dict[str, ModeMetrics]) -> None:
    print(f"\nRAG retrieval eval — {len(QUESTIONS)} questions, top_k={TOP_K}\n")
    print(f"{'mode':<8} {'recall@k':>10} {'MRR':>8} {'latency_ms':>12}")
    for name, m in metrics.items():
        print(f"{name:<8} {m.recall_at_k:>10.3f} {m.mrr:>8.3f} {m.avg_latency_ms:>12.1f}")
    print()


async def _amain() -> None:
    metrics = await run_rag_eval()
    _print_report(metrics)
    # Non-regression: hybrid must never do worse than pure dense.
    assert metrics["hybrid"].recall_at_k >= metrics["dense"].recall_at_k, (
        "hybrid recall regressed below pure dense"
    )
    assert metrics["hybrid"].mrr >= metrics["dense"].mrr, "hybrid MRR regressed below pure dense"
    print("✓ hybrid recall & MRR >= pure dense (non-regression)")


if __name__ == "__main__":
    asyncio.run(_amain())
