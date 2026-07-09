# Phase 2 — RAG baseline

Recorded baseline for the hybrid RAG pipeline (Phase 2 DoD). Re-run any time
with `uv run python scripts/eval_run.py --suite rag`.

## Environment

- Reference hardware: MacBook M4, 24 GB, `DEPLOY_PROFILE=split`.
- Dense embeddings: `bge-m3` (1024-dim) via Ollama.
- Sparse embeddings: `Qdrant/bm42-all-minilm-l6-v2-attentions` via FastEmbed.
- Fusion: Reciprocal Rank Fusion (RRF), server-side in Qdrant.
- Reranker: `bge-reranker-v2-m3`, lazy-loaded native MPS service on `:8081`.
  **Not running in this baseline run** — rerank degrades gracefully to a no-op
  (fused RRF order preserved). Add its latency on top when the service is up.

## Retrieval quality (eval suite, 25 questions, top_k=5)

| mode   | recall@k | MRR   |
|--------|----------|-------|
| hybrid | 1.000    | 1.000 |
| dense  | 1.000    | 1.000 |

The reference embedder `bge-m3` is strong enough to **saturate** recall on this
small, clean corpus — it ranks even bare-keyword and opaque-identifier queries
correctly, so pure dense already scores at the ceiling. The suite therefore
asserts **non-regression** (hybrid ≥ dense on recall@k and MRR). Hybrid's strict
advantage (robust exact-term / keyword lookup) materialises on larger, noisier
corpora: seed your own 50-100 documents with `scripts/seed_rag.py` and re-run.

## Latency (retrieval + fusion, no rerank)

- Cold query (novel text, dense embedding computed via Ollama): **~80 ms**.
- Warm query (embedding served from the Redis cache): **~7 ms**.

Measured end-to-end over `/v1/rag/query`. Cross-encoder rerank adds its own
cost when the MPS service is running; it is excluded here.
