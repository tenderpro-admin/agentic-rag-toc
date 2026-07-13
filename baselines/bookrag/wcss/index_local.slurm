#!/bin/bash
#SBATCH --job-name=bookrag-index-local
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=lem-gpu
# Account/partition: export SBATCH_ACCOUNT=<your-grant> (and optionally SBATCH_PARTITION) before submitting; sbatch reads them from the environment.
#SBATCH --gres=gpu:hopper:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb
#SBATCH --time=24:00:00
#
# OFFICIAL BookRAG local-model fleet on one 4x H100 node: stand up the vLLM +
# sglang servers, then build the BookIndex (MinerU vlm parse -> tree -> graph ->
# vdb) entirely on-cluster, no OpenAI. run_remote.sh sets WORKDIR/NUM/NSPLIT.
set -uo pipefail
: "${WORKDIR:?set WORKDIR via --export}"
export COMMAND="index"
export STAGE="${STAGE:-all}"
export NUM="${NUM:-1}"
export NSPLIT="${NSPLIT:-1}"
export CONFIG="config/financebench_wcss_local.yaml"
bash "$WORKDIR/wcss/serve_fleet.sh"
