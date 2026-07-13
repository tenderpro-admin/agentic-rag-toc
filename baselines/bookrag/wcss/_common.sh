#!/bin/bash
# Shared setup sourced by every BookRAG .slurm script on WCSS (Lem GPU / H100).
#
# Sourced — NOT executed. The .slurm caller sets task vars (CONFIG, DATASET_CFG,
# COMMAND, STAGE, NUM, NSPLIT, MINERU_DEVICE) then `source "$WORKDIR/wcss/_common.sh"`
# and calls `bookrag_run`.
#
# Per the WCSS skill:
#   - use a WORKDIR env var, never ${BASH_SOURCE[0]} (SLURM copies scripts to its
#     tmpdir, so BASH_SOURCE resolves to the tmpdir copy, not the repo).
#   - every file path handed to the job is ABSOLUTE.
#   - per-job ephemeral caches in $TMPDIR; the big model cache is PERSISTENT and
#     pre-populated on the login node (GPU compute nodes are treated as offline).
set -euo pipefail

# --- WORKDIR: the BookRAG checkout on shared home (passed by the .slurm) -------
: "${WORKDIR:?WORKDIR must point to the BookRAG checkout, e.g. ~/projects/BookRAG}"
cd "$WORKDIR"

# --- conda env (built once by wcss/setup_env.sh on the login node) ------------
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bookrag}"
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- secrets: OPENAI_API_KEY from .env ONLY (gitignored) ----------------------
if [[ -f "$WORKDIR/.env" ]]; then
  set -a; . "$WORKDIR/.env"; set +a
fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY missing — put it in $WORKDIR/.env}"

# --- caches ------------------------------------------------------------------
# Ephemeral, per-job (avoids the shared-$HOME cache-corruption trap):
export PIP_CACHE_DIR="$TMPDIR/pip"
export XDG_CACHE_HOME="$TMPDIR/cache"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Persistent model cache, pre-populated on the login node (Qwen reranker, MinerU
# PDF-Extract-Kit, spaCy). Compute nodes may have no internet -> run HF offline.
export HF_HOME="${HF_HOME:-$WORKDIR/.cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$WORKDIR/.cache/modelscope}"
export MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-local}"
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE"

# --- runtime knobs -----------------------------------------------------------
export PYTHONPATH="$WORKDIR"
export MINERU_DEVICE_MODE="${MINERU_DEVICE:-cuda}"
export PYTORCH_ENABLE_MPS_FALLBACK=1   # harmless on cuda; keeps parity with mac
export TOKENIZERS_PARALLELISM=false

# --- dataset cfg with ABSOLUTE paths -----------------------------------------
# The upstream Scripts/cfg/financebench.yaml uses `../datasets/...` /`../runs/...`,
# which only resolves when CWD=Scripts/. We run from the repo root, and the WCSS
# skill is explicit: every config path must be absolute. Generate a per-job cfg
# (paths are host-specific via $WORKDIR) and point the run at it.
DATASET_CFG_ABS="$TMPDIR/financebench_dataset.yaml"
cat > "$DATASET_CFG_ABS" <<EOF
dataset_path: $WORKDIR/datasets/financebench.json
working_dir: $WORKDIR/runs/financebench
dataset_name: financebench
EOF
DATASET_CFG="$DATASET_CFG_ABS"

echo "================================================================"
echo " BookRAG WCSS job"
echo "   node=$(hostname)  job=${SLURM_JOB_ID:-NA}  part=${SLURM_JOB_PARTITION:-NA}"
echo "   WORKDIR=$WORKDIR  env=$CONDA_ENV  python=$(which python)"
echo "   MINERU_DEVICE_MODE=$MINERU_DEVICE_MODE  HF_HOME=$HF_HOME"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || true

# --- input validation (fail fast, before anything expensive) -----------------
bookrag_validate() {
  : "${CONFIG:?CONFIG (system config) not set}"
  : "${COMMAND:?COMMAND must be 'index' or 'rag'}"
  # CONFIG may be relative to the repo root or an absolute path (e.g. a patched
  # copy in $TMPDIR); accept either.
  local cfg="$CONFIG"; [[ "$cfg" = /* ]] || cfg="$WORKDIR/$cfg"
  [[ -f "$cfg" ]]               || { echo "missing config: $cfg"; exit 1; }
  [[ -f "$DATASET_CFG" ]]       || { echo "missing dataset cfg: $DATASET_CFG"; exit 1; }
  # The real failure last time: the dataset JSON itself was absent / mis-pathed.
  [[ -s "$WORKDIR/datasets/financebench.json" ]] || {
    echo "missing/empty dataset: $WORKDIR/datasets/financebench.json (build it first)"; exit 1; }
}

# --- run BookRAG main.py -----------------------------------------------------
bookrag_run() {
  bookrag_validate
  local stage_args=()
  if [[ "$COMMAND" == "index" ]]; then
    stage_args=(index --stage "${STAGE:-all}")
  else
    stage_args=(rag)
  fi
  local err_marker="$TMPDIR/.bookrag_run_start"
  touch "$err_marker"
  echo "+ python main.py -c $CONFIG -d $DATASET_CFG --num ${NUM:-1} --nsplit ${NSPLIT:-1} ${stage_args[*]}"
  python main.py \
    -c "$CONFIG" \
    -d "$DATASET_CFG" \
    --num "${NUM:-1}" --nsplit "${NSPLIT:-1}" \
    "${stage_args[@]}"

  # main.py logs some failures (e.g. dataset-not-found, reranker crash) yet still
  # exits 0 — catch its own error-report file written during THIS run.
  if find "$WORKDIR" -maxdepth 1 -name 'financebench-index_error*.txt' \
        -newer "$err_marker" 2>/dev/null | grep -q .; then
    echo "ERROR: BookRAG wrote a fresh index error report:"
    find "$WORKDIR" -maxdepth 1 -name 'financebench-index_error*.txt' -newer "$err_marker" \
      -exec tail -n 20 {} \;
    exit 1
  fi

  # also assert the expected artifacts exist so a silent failure can't pass.
  if [[ "$COMMAND" == "index" ]]; then
    if ! ls "$WORKDIR/runs/financebench"/*/ >/dev/null 2>&1; then
      echo "ERROR: index produced no runs/financebench/<doc>/ output"; exit 1
    fi
  else
    # BookRAG writes final_results.json under <doc_uuid>/eval_<dataset>_<strategy>/,
    # not directly under <doc_uuid> — search recursively.
    if ! find "$WORKDIR/runs/financebench" -name final_results.json 2>/dev/null | grep -q .; then
      echo "ERROR: rag produced no final_results.json"; exit 1
    fi
  fi
}
