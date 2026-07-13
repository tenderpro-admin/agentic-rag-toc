#!/bin/bash
#SBATCH --job-name=bookrag-ablation
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=lem-gpu
# Account/partition: export SBATCH_ACCOUNT=<your-grant> (and optionally SBATCH_PARTITION) before submitting; sbatch reads them from the environment.
#SBATCH --gres=gpu:hopper:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb
#SBATCH --time=12:00:00
#
# WHOLE answer-model ablation in ONE on-cluster job — no local<->login ssh per
# stage. Brings the embed vLLM server up ONCE, then for each model: clears prior
# RAG outputs (force-fresh), patches the LLM, answers over the FROZEN index, and
# exports predictions (with litellm $cost) to results/financebench/bookrag/<model>/.
# It does NOT judge: judging is done LOCALLY afterward by wcss/judge_local.sh (the
# arag-toc gpt-5.4-mini judge). Reranker loads in-process per rag.
#
# Submit (from the WCSS login node):
#   cd ~/projects/BookRAG && sbatch --export=ALL,WORKDIR=$HOME/projects/BookRAG,\
#     MODELS="gpt-5-mini-2025-08-07 gpt-5-2025-08-07 gpt-4.1-mini-2025-04-14 \
#     gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18" wcss/ablation.slurm
set -uo pipefail
: "${WORKDIR:?set WORKDIR}"
MODELS="${MODELS:?set MODELS (space-separated answer models)}"
DOCS="${DOCS:-BOEING_2022_10K}"

export COMMAND="rag" CONFIG="config/financebench_wcss_gpt.yaml" CLOUD_LLM=1
source "$WORKDIR/wcss/_common.sh"   # conda, .env (OPENAI_API_KEY), caches, dataset cfg

# Dataset must match the frozen index docs.
FINANCEBENCH_DIR="${FINANCEBENCH_DIR:-$HOME/finance_bench_root}"
FINANCEBENCH_DIR="$FINANCEBENCH_DIR" python3 Scripts/preprocess/financebench_to_bookrag.py \
  --docs $DOCS --out datasets/financebench.json
# Optional smoke cap: NQ=N truncates the dataset to the first N questions so a
# quick single-model run finishes fast (validates the model actually answers).
if [ -n "${NQ:-}" ]; then
  python - "$WORKDIR/datasets/financebench.json" "$NQ" <<'PY'
import json, sys
p, n = sys.argv[1], int(sys.argv[2])
d = json.load(open(p))
json.dump(d[:n], open(p, "w"))
print(f">> NQ cap: dataset truncated to {min(n,len(d))} questions")
PY
fi

# --- embed vLLM server (Qwen3-Embedding-0.6B), once -------------------------
source "$WORKDIR/wcss/fleet_paths.env"   # EMBED_PATH
VLLM_SIF="/lustre/software-data/container-images/vllm-gpu-v0.9.0.1.sif"
command -v apptainer >/dev/null 2>&1 || source /usr/local/sbin/modules.sh 2>/dev/null || true
LOGDIR="$WORKDIR/wcss/logs/ablation-${SLURM_JOB_ID:-local}"; mkdir -p "$LOGDIR"
ec="$TMPDIR/vllm-embed"; mkdir -p "$ec"/{cache,triton,vllm,outlines,numba}
# .cache may be a symlink -> /lustre/...; apptainer does not expose the symlink
# TARGET inside the container, so bind the resolved real path too (else the embed
# model dir is a dangling symlink -> "Invalid local directory" and :8007 never UP).
CREAL="$(readlink -f "$WORKDIR/.cache" 2>/dev/null)"
CACHE_BIND=""; [ -n "$CREAL" ] && [ "$CREAL" != "$WORKDIR/.cache" ] && CACHE_BIND="--bind $CREAL:$CREAL"
apptainer exec --nv --bind "$WORKDIR/.cache:$WORKDIR/.cache" --bind "$TMPDIR:$TMPDIR" $CACHE_BIND \
  --env CUDA_VISIBLE_DEVICES=2 --env PYTHONUNBUFFERED=1 --env HOME="$ec" \
  --env XDG_CACHE_HOME="$ec/cache" --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
  --env VLLM_CACHE_ROOT="$ec/vllm" --env TRITON_CACHE_DIR="$ec/triton" \
  --env OUTLINES_CACHE_DIR="$ec/outlines" --env NUMBA_CACHE_DIR="$ec/numba" \
  "$VLLM_SIF" vllm serve "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-0.6B \
  --task embed --gpu-memory-utilization 0.18 --enforce-eager --host 127.0.0.1 --port 8007 \
  > "$LOGDIR/embed.log" 2>&1 &
EMBED_PID=$!
trap 'kill $EMBED_PID 2>/dev/null' EXIT
echo -n ">> waiting for embed "; EMBED_UP=0; for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8007/health >/dev/null 2>&1 && { echo " UP (${i}0s)"; EMBED_UP=1; break; }
  sleep 5; echo -n .; done
# Fatal: without a healthy embedder every model's rag would fail silently.
[[ "$EMBED_UP" == "1" ]] || { echo " TIMEOUT — embed server never became healthy"; tail -n 30 "$LOGDIR/embed.log" 2>/dev/null; exit 1; }

export MINERU_DEVICE_MODE="cuda:3"   # reranker in-process on cuda:3
# Force modelscope/HF offline: the in-process reranker reloads per doc and
# pings www.modelscope.cn to resolve the model id even when cached -> 60s
# timeouts that silently kill a doc's inference. local_files_only is ignored by
# modelscope, so we ALSO rewrite the reranker model_name to its absolute local
# snapshot path below (a real dir path skips hub resolution). Belt-and-suspenders.
export MODELSCOPE_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# tiktoken downloads its BPE encodings (cl100k_base/o200k_base) from
# openaipublic.blob.core.windows.net on first use; compute nodes can't reach it
# -> inference fails. Pre-warmed cache lives in .cache/tiktoken (fetched on the
# login node which HAS internet); point tiktoken at it so it loads offline.
export TIKTOKEN_CACHE_DIR="$WORKDIR/.cache/tiktoken"
RERANK_LOCAL="$WORKDIR/.cache/modelscope/models/Qwen/Qwen3-Reranker-0.6B"

# --- per-model: force-fresh rag -> export(cost) -----------------------------
FAILED=""
for M in $MODELS; do
  echo "==================== $M ===================="
  rm -rf "$WORKDIR/runs/financebench"/*/eval_* 2>/dev/null || true
  # per-model config: gpt config + this LLM + real api key (tmpfs)
  CFG="$TMPDIR/cfg_$M.yaml"
  python - "$WORKDIR/$CONFIG" "$CFG" "$M" "$RERANK_LOCAL" <<'PY'
import sys, os, yaml
c = yaml.safe_load(open(sys.argv[1]))
llm = c.setdefault("llm", {}); llm["model_name"] = sys.argv[3]
llm["api_key"] = os.environ["OPENAI_API_KEY"]
rerank_local = sys.argv[4]
# Point every reranker_config at the absolute local snapshot path so modelscope
# from_pretrained loads from disk (no www.modelscope.cn hub ping / timeout).
def fix(o):
    if isinstance(o, dict):
        if o.get("model_name") == "Qwen/Qwen3-Reranker-0.6B":
            o["model_name"] = rerank_local
        for v in o.values(): fix(v)
    elif isinstance(o, list):
        for v in o: fix(v)
fix(c)
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
PY
  # rag + export only (real per-model tokens + litellm $cost). Judging happens
  # LOCALLY afterward (the arag-toc gpt-5-mini judge), so no dependency on
  # arag-toc being present on the cluster. Nothing here may abort the loop.
  if ! python main.py -c "$CFG" -d "$DATASET_CFG" --num 1 --nsplit 1 rag; then
    echo ">> rag FAILED for $M"; FAILED="$FAILED $M"; continue
  fi
  if ! python -m Eval.export_predictions --runs-dir runs/financebench \
       --model "$M" --out "results/financebench/bookrag/$M/predictions.json"; then
    echo ">> export FAILED for $M"; FAILED="$FAILED $M"; continue
  fi
  echo ">> $M done (predictions exported)"
done
if [[ -n "${FAILED# }" ]]; then
  echo "ABLATION_DONE rc=1 — FAILED models:$FAILED at $(date)"; exit 1
fi
echo "ABLATION_DONE rc=0 at $(date)"
