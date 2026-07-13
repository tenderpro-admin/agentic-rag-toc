#!/bin/bash
# One-time BookRAG environment build on the WCSS LOGIN node (which has internet).
# GPU compute nodes are treated as offline, so EVERY model is pre-fetched here
# into a persistent cache that the .slurm jobs read with HF_HUB_OFFLINE=1.
#
#   ssh <your-login>@ui.wcss.pl 'cd ~/projects/BookRAG && bash wcss/setup_env.sh'
#
# Idempotent: re-running re-uses the conda env and skips already-downloaded models.
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/projects/BookRAG}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bookrag}"
export HF_HOME="$WORKDIR/.cache/hf"
export MODELSCOPE_CACHE="$WORKDIR/.cache/modelscope"
export MINERU_MODEL_SOURCE=huggingface
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE"
cd "$WORKDIR"

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if ! conda env list | grep -q "/envs/$CONDA_ENV"; then
  echo ">> creating conda env $CONDA_ENV (python 3.12)"
  conda create -n "$CONDA_ENV" python=3.12 -y
fi
conda activate "$CONDA_ENV"
python -V

echo ">> pip install BookRAG requirements (torch 2.7.1 cuda, transformers, mineru...)"
pip install --upgrade pip
pip install -r requirements.txt
# Deps not pinned in requirements.txt but required for the MinerU 'pipeline'
# backend + tree stage (surfaced one-by-one as 'No module named X'):
pip install 'mineru[core]==2.1.11' ftfy dill

echo ">> pre-download MinerU pipeline models (offline use on compute nodes)"
mineru-models-download -s huggingface -m pipeline || \
  echo "WARN: mineru-models-download returned non-zero; check logs above"

echo ">> pre-download Qwen3-Reranker-0.6B into $HF_HOME"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-Reranker-0.6B")
print("reranker cached")
PY

echo ">> smoke-import the heavy libs"
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
import mineru  # noqa
print("mineru import OK")
PY

echo "SETUP_DONE — env=$CONDA_ENV  HF_HOME=$HF_HOME"
