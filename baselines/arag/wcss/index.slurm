#!/usr/bin/env bash
#SBATCH --job-name=arag-index
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Account/partition: export SBATCH_ACCOUNT=<your-grant> (and optionally SBATCH_PARTITION) before submitting; sbatch reads them from the environment.
# Build the FROZEN per-doc Qwen embedding index for all FinanceBench docs.
# Embed-only (no graph/tree) -> fast. Idempotent: skips docs already indexed.
set -euo pipefail
cd "${WORKDIR:-$HOME/projects/arag}"
source wcss/_common.sh

# SLURM logs to lustre (home quota is tight).
DOCS="${DOCS:-all}"

echo ">> convert: MinerU markdown -> per-doc A-RAG passages ($DOCS)"
python financebench/financebench_to_arag.py \
    --docs $DOCS \
    --runs-dir "$RUNS" \
    --out "$ARAG_DATA" \
    --data-root "${FINANCEBENCH_DIR:-$HOME/finance_bench_root}"

echo ">> index: Qwen3-Embedding-0.6B on cuda:0 (frozen artifact)"
python financebench/financebench_run.py \
    --data-dir "$ARAG_DATA" \
    --index-only \
    --embed-model "$EMBED_MODEL" \
    --device cuda:0 \
    --batch-size 64

N=$(find "$ARAG_DATA" -name sentence_index.pkl | wc -l)
echo "INDEX_DONE indexes=$N"
