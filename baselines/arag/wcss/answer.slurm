#!/usr/bin/env bash
#SBATCH --job-name=arag-answer
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Account/partition: export SBATCH_ACCOUNT=<your-grant> (and optionally SBATCH_PARTITION) before submitting; sbatch reads them from the environment.
# Answer all FinanceBench questions over the FROZEN per-doc index with ONE
# matrix model (gpt-* via OpenAI cloud). GPU is used for Qwen query-embedding
# only. Resume-safe (skips qids already in predictions.jsonl). Then export to
# the arag-toc predictions schema for the shared judge.
set -euo pipefail
cd "${WORKDIR:-$HOME/projects/arag}"
source wcss/_common.sh

MODEL="${MODEL:?set MODEL=<answer model id>}"
OUT_DIR="$RESULTS_DIR/$MODEL"
mkdir -p "$OUT_DIR"

# LIMIT set => smoke: separate preds file, no export (don't pollute full resume).
LIMIT_FLAG=""; PREDS="$OUT_DIR/predictions.jsonl"
if [ -n "${LIMIT:-}" ]; then LIMIT_FLAG="--limit $LIMIT"; PREDS="$OUT_DIR/predictions_smoke.jsonl"; fi

# Reasoning models need reasoning_effort; chat models must NOT receive it.
EFFORT_FLAG=""
case "$MODEL" in
  gpt-5*|o1*|o3*|o4*) EFFORT_FLAG="--reasoning-effort medium" ;;
esac

echo ">> answer: $MODEL over frozen index ${LIMIT:+(smoke limit=$LIMIT)}"
python financebench/financebench_run.py \
    --data-dir "$ARAG_DATA" \
    --out "$PREDS" \
    --embed-model "$EMBED_MODEL" \
    --device cuda:0 \
    --model "$MODEL" \
    --max-loops 15 \
    $LIMIT_FLAG $EFFORT_FLAG

if [ -z "${LIMIT:-}" ]; then
  echo ">> export -> arag-toc schema"
  python financebench/export_predictions.py \
      --preds "$PREDS" --model "$MODEL" --out "$OUT_DIR/predictions.json"
fi

N=$(grep -c . "$PREDS" || true)
echo "ANSWER_DONE model=$MODEL preds=$N smoke=${LIMIT:-0}"
