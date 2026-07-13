#!/usr/bin/env bash
# Shared env for A-RAG FinanceBench jobs on WCSS. Source this; do not exec.
# Activates the conda env, sets caches off the home quota, loads the OpenAI key,
# and exports the canonical paths used by index/answer scripts.
set -euo pipefail

: "${WORKDIR:=$HOME/projects/arag}"                       # A-RAG repo on WCSS
BOOKRAG="${BOOKRAG:-$HOME/projects/BookRAG}"               # MinerU markdown source
RUNS="${RUNS:-$BOOKRAG/runs/financebench}"                 # <uuid>/auto/<doc>.md
EMBED_MODEL="${EMBED_MODEL:-$BOOKRAG/.cache/modelscope/models/Qwen/Qwen3-Embedding-0.6B}"
# Per-doc corpus + frozen index. Home (~16G free) not lustre: the lustre group
# allocation is over its default quota (BookRAG runs+cache), and A-RAG's index is
# small (~2-3G for 84 docs). Override ARAG_DATA to relocate if home tightens.
ARAG_DATA="${ARAG_DATA:-$WORKDIR/data/financebench}"
RESULTS_DIR="${RESULTS_DIR:-$WORKDIR/results/financebench/arag}"

# Conda env (reuse BookRAG's; has torch+cu126). sentence-transformers added by setup.sh.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bookrag

# Caches off the 50G home quota; compute nodes have flaky internet -> all local.
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$BOOKRAG/.cache/tiktoken}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export HF_HOME="${HF_HOME:-$BOOKRAG/.cache/hf}"
export TOKENIZERS_PARALLELISM=false

# OpenAI key (answer models are cloud gpt-*). Reuse BookRAG's .env.
set -a; . "$BOOKRAG/.env"; set +a
export ARAG_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY missing in $BOOKRAG/.env}"
export ARAG_BASE_URL="${ARAG_BASE_URL:-https://api.openai.com/v1}"

export PYTHONPATH="$WORKDIR/src:$WORKDIR"
mkdir -p "$ARAG_DATA" "$RESULTS_DIR"
