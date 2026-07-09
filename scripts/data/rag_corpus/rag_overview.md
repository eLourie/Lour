# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a retriever with a generator. The
retriever finds relevant passages from a knowledge base, and the language model
conditions its answer on those passages instead of relying only on parametric
memory.

## Hybrid search

Hybrid search fuses dense vector similarity with sparse lexical matching. Dense
retrieval captures meaning; sparse methods such as BM25 or BM42 capture exact
terms, rare tokens, and identifiers. Reciprocal Rank Fusion (RRF) merges the two
ranked lists into one.

## Reranking

A cross-encoder reranker scores each query-passage pair jointly and reorders the
fused candidates. It is more accurate than bi-encoder similarity but slower, so
it typically runs only on the top candidates.
