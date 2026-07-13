#!/bin/bash
#SBATCH --job-name=bookrag-rag-local
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=lem-gpu
#SBATCH --account=hpc-tkajdanowicz-1763478893
#SBATCH --gres=gpu:hopper:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb
#SBATCH --time=12:00:00
#
# Answer FinanceBench over the frozen BookIndex with the official local fleet,
# then export predictions in the shared schema (done inside serve_fleet.sh on
# rag). run_remote.sh sets WORKDIR/NUM/NSPLIT/ANSWER_MODEL.
set -uo pipefail
: "${WORKDIR:?set WORKDIR via --export}"
export COMMAND="rag"
export NUM="${NUM:-1}"
export NSPLIT="${NSPLIT:-1}"
# CONFIG/CLOUD_LLM overridable for answer-model ablations: default = all-local
# Qwen fleet; for a cloud answer LLM pass CONFIG=config/financebench_wcss_gpt.yaml
# CLOUD_LLM=1 (skips the Qwen LLM server; embed+reranker unchanged).
export CONFIG="${CONFIG:-config/financebench_wcss_local.yaml}"
export CLOUD_LLM="${CLOUD_LLM:-0}"
export ANSWER_MODEL="${ANSWER_MODEL:-Qwen3-8B-AWQ}"
bash "$WORKDIR/wcss/serve_fleet.sh"
