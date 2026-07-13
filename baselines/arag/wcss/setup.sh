#!/usr/bin/env bash
# One-time setup on the WCSS LOGIN node (has internet; compute nodes do not).
# Adds sentence-transformers to the bookrag env and pre-warms the tiktoken cache.
set -euo pipefail
BOOKRAG="${BOOKRAG:-$HOME/projects/BookRAG}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bookrag

# A-RAG semantic_search/build_index need sentence-transformers; bookrag env has
# torch+transformers already, so this is additive. (Login node only — internet.)
pip install --quiet "sentence-transformers>=2.2.0"

# tiktoken lazy-downloads o200k/cl100k from the internet on first use; compute
# nodes can't reach it. Pre-warm into a cache dir the jobs point at.
export TIKTOKEN_CACHE_DIR="$BOOKRAG/.cache/tiktoken"
mkdir -p "$TIKTOKEN_CACHE_DIR"
python - <<'PY'
import tiktoken
tiktoken.encoding_for_model("gpt-4o")     # o200k_base
tiktoken.get_encoding("cl100k_base")
print("tiktoken cache warmed")
PY

# Verify the Qwen embedder loads from the local snapshot (offline).
EMBED="$BOOKRAG/.cache/modelscope/models/Qwen/Qwen3-Embedding-0.6B"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - "$EMBED" <<'PY'
import sys
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(sys.argv[1], device="cpu")
print("embed OK dim:", m.get_sentence_embedding_dimension())
PY
echo "SETUP_DONE"
