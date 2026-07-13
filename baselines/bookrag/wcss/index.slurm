#!/bin/bash
#SBATCH --job-name=bookrag-index
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=lem-gpu
# Account/partition: export SBATCH_ACCOUNT=<your-grant> (and optionally SBATCH_PARTITION) before submitting; sbatch reads them from the environment.
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#
# Build the frozen BookIndex (MinerU parse on GPU -> tree -> graph -> vdb) for the
# FinanceBench docs in the dataset. MinerU + the Qwen reranker run on the H100;
# the LLM/VLM/embeddings are OpenAI cloud. Override knobs at submit time, e.g.:
#   sbatch --export=ALL,WORKDIR=$HOME/projects/BookRAG,NUM=1,NSPLIT=1 wcss/index.slurm
# run_remote.sh sets these for you.

set -euo pipefail
: "${WORKDIR:?set WORKDIR via --export}"

# Task-specific vars (overridable via --export); abs paths resolved in _common.sh.
export CONFIG="${CONFIG:-config/financebench_wcss.yaml}"
export DATASET_CFG="${DATASET_CFG:-Scripts/cfg/financebench.yaml}"
export COMMAND="index"
export STAGE="${STAGE:-all}"
export NUM="${NUM:-1}"
export NSPLIT="${NSPLIT:-1}"
export MINERU_DEVICE="${MINERU_DEVICE:-cuda}"

source "$WORKDIR/wcss/_common.sh"
bookrag_run
# Under `set -e` reaching this line means the index build succeeded (a failure
# would have exited the job non-zero), so there is no rc to report here.
echo "INDEX_DONE at $(date)"
