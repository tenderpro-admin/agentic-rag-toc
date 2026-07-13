#!/bin/bash
# One-time: pre-download BookRAG's official local Qwen fleet from MODELSCOPE on
# the WCSS login node. We use ModelScope, not HuggingFace: the HF route from WCSS
# is throttled to ~6 KB/s, while modelscope.cn pulls these (Alibaba) models at
# tens of MB/s. Writes wcss/fleet_paths.env mapping each role to its local path
# so serve_fleet.sh points vLLM / BookRAG straight at the cached dirs (offline).
#
#   ssh <your-login>@ui.wcss.pl 'cd ~/projects/BookRAG && bash wcss/setup_fleet.sh'
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/projects/BookRAG}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bookrag}"
export MODELSCOPE_CACHE="$WORKDIR/.cache/modelscope"
mkdir -p "$MODELSCOPE_CACHE"
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$WORKDIR"

MANIFEST="$WORKDIR/wcss/fleet_paths.env"
: > "$MANIFEST"

# role  modelscope_id
download() {  # var_name modelscope_id
  local var="$1" id="$2"
  echo ">> [$var] modelscope: $id"
  local path
  # modelscope's snapshot_download prints progress chatter to stdout, so the real
  # path is the LAST line only.
  path=$(python - "$id" <<'PY' | tail -1
import sys
from modelscope.hub.snapshot_download import snapshot_download
print(snapshot_download(sys.argv[1]))
PY
)
  echo "$var=$path" >> "$MANIFEST"
  echo "   -> $path"
}

# Small, reliable models first so the core RAG fleet is ready fast; the big VLM
# (16 GB, flaky on the WCSS↔ModelScope link) goes LAST so a VLM stall can't block
# the others. serve_fleet.sh runs VLM-optional, so a missing VLM_PATH still works.
download LLM_PATH    "Qwen/Qwen3-8B-AWQ"
download EMBED_PATH  "Qwen/Qwen3-Embedding-0.6B"
download RERANK_PATH "Qwen/Qwen3-Reranker-4B"
download GME_PATH    "iic/gme-Qwen2-VL-2B-Instruct"
download VLM_PATH    "Qwen/Qwen2.5-VL-7B-Instruct" || echo "WARN: VLM download failed; fleet runs VLM-optional"

echo "=== fleet_paths.env ==="
cat "$MANIFEST"
echo "SETUP_FLEET_DONE — manifest=$MANIFEST"
