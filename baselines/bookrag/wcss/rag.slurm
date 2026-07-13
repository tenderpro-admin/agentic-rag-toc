#!/bin/bash
#SBATCH --job-name=bookrag-rag
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=lem-gpu
# Account/partition: export SBATCH_ACCOUNT=<your-grant> (and optionally SBATCH_PARTITION) before submitting; sbatch reads them from the environment.
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#
# Answer FinanceBench questions over the frozen BookIndex (RAG) and export the
# shared predictions JSON. The reranker/embeddings used at query time match the
# ones the index was built with, so RAG also runs on the H100. The exported
# predictions.json is then judged by the SAME gpt-5-mini judge as PageIndex
# (in the arag-toc repo) — apples-to-apples FinanceBench comparison.

set -euo pipefail
: "${WORKDIR:?set WORKDIR via --export}"

export CONFIG="${CONFIG:-config/financebench_wcss.yaml}"
export DATASET_CFG="${DATASET_CFG:-Scripts/cfg/financebench.yaml}"
export COMMAND="rag"
export NUM="${NUM:-1}"
export NSPLIT="${NSPLIT:-1}"
export MINERU_DEVICE="${MINERU_DEVICE:-cuda}"
ANSWER_MODEL="${ANSWER_MODEL:-gpt-4o-mini}"
RUNS_DIR="${RUNS_DIR:-runs/financebench}"
PRED_OUT="${PRED_OUT:-results/financebench/bookrag/$ANSWER_MODEL/predictions.json}"

source "$WORKDIR/wcss/_common.sh"
bookrag_run

# Flatten BookRAG per-doc outputs into the shared predictions schema.
echo "+ export_predictions -> $PRED_OUT"
python -m Eval.export_predictions \
  --runs-dir "$RUNS_DIR" --model "$ANSWER_MODEL" --out "$PRED_OUT"
# Under `set -e` reaching this line means every step above succeeded (a failure
# would have exited the job non-zero), so there is no rc to report here.
echo "RAG_DONE at $(date)"
