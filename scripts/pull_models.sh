#!/usr/bin/env bash
# =============================================================================
#  Pull required Ollama models for the reference M4 24GB Split profile.
#  Run this once after `make up` to download models onto the host.
#
#  Usage: bash scripts/pull_models.sh
#         OLLAMA_BASE_URL=http://localhost:11434 bash scripts/pull_models.sh
# =============================================================================
set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

MODELS=(
  "qwen3:14b"   # main LLM: routing + planning + tool-calling
  "bge-m3"      # dense embeddings (1024 dim, multilingual)
)

echo "→ Pulling Ollama models from ${OLLAMA_BASE_URL}"
echo "  Note: qwen3:14b ≈ 9 GB, bge-m3 ≈ 2.3 GB — be patient on first run."
echo ""

for model in "${MODELS[@]}"; do
  echo "  • Pulling ${model} ..."
  ollama pull "${model}" 2>&1 | tail -1
  echo "    ✓ ${model}"
done

echo ""
echo "✓ All models ready."
echo ""
echo "Loaded models:"
ollama list
