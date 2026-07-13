#!/bin/bash
# Local driver: run a BookRAG SLURM job on WCSS and bring the artifacts back.
#
#   push (rsync) -> sbatch --wait (blocks until the job finishes) -> pull (rsync)
#
# Designed to be the command behind a DVC stage: it is synchronous (sbatch --wait)
# and its exit code is the job's exit code, so `dvc repro` fails if the job fails.
#
# Usage:
#   wcss/run_remote.sh index            # build the frozen BookIndex on the H100
#   wcss/run_remote.sh rag              # answer + export predictions on the H100
#
# Env (override as needed):
#   WCSS_HOST     ssh target            (default <your-login>@ui.wcss.pl)
#   WCSS_REMOTE   remote checkout path  (default projects/BookRAG, relative to $HOME)
#   NUM,NSPLIT    BookRAG split control (default 1 / 1)
#   ANSWER_MODEL  label for rag export  (default gpt-4o-mini)
#   SMOKE=1       use the short queue + a 6h wall clock (single-doc smoke)
#   SBATCH_EXTRA  extra sbatch flags    (e.g. "--time=24:00:00 --partition=lem-gpu")
set -euo pipefail

CMD="${1:?usage: run_remote.sh <index|rag>}"
case "$CMD" in index|rag) ;; *) echo "command must be index|rag"; exit 2;; esac

WCSS_HOST="${WCSS_HOST:-<your-login>@ui.wcss.pl}"
WCSS_REMOTE="${WCSS_REMOTE:-projects/BookRAG}"   # relative to remote $HOME
FINANCEBENCH_DIR="${FINANCEBENCH_DIR:-\$HOME/finance_bench_root}"  # remote data root
DOCS="${DOCS:-all}"                              # FinanceBench docs to index
NUM="${NUM:-1}"; NSPLIT="${NSPLIT:-1}"
# FLEET=local (default) = official Qwen fleet on 4 H100 (index_local.slurm);
# FLEET=cloud = the gpt-4o-mini single-GPU path (index.slurm).
FLEET="${FLEET:-local}"
if [[ "$FLEET" == "local" ]]; then
  SLURM_FILE="wcss/${CMD}_local.slurm"
  ANSWER_MODEL="${ANSWER_MODEL:-Qwen3-8B-AWQ}"
else
  SLURM_FILE="wcss/${CMD}.slurm"
  ANSWER_MODEL="${ANSWER_MODEL:-gpt-4o-mini}"
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # BookRAG repo root

SBATCH_FLAGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  SBATCH_FLAGS+=(--partition=lem-gpu-short --time=06:00:00)
fi
# shellcheck disable=SC2206
[[ -n "${SBATCH_EXTRA:-}" ]] && SBATCH_FLAGS+=(${SBATCH_EXTRA})

# The WCSS login node is frequently congested — ssh/rsync can hang or drop. Use
# short connect timeouts + keepalive, and retry every remote op so a transient
# stall never wedges `dvc repro`. The long-running job is decoupled from any single
# ssh: we submit (quick), then POLL (short, retryable), instead of `sbatch --wait`.
SSH_OPTS="-o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 -o BatchMode=yes"

ssh_r() {  # retryable ssh; prints remote stdout, returns 0 on success
  local t out
  for t in 1 2 3 4 5 6 7 8; do
    if out=$(timeout 90 ssh $SSH_OPTS "$WCSS_HOST" "$1" 2>/dev/null); then
      printf '%s' "$out"; return 0
    fi
    sleep 20
  done
  return 1
}
rsync_r() {  # retryable rsync ($@ = rsync args after -e)
  local t
  for t in 1 2 3 4 5 6; do
    timeout 180 rsync -az -e "ssh $SSH_OPTS" "$@" && return 0
    sleep 20
  done
  return 1
}

echo ">> [1/3] push $HERE -> $WCSS_HOST:~/$WCSS_REMOTE"
rsync_r --delete \
  --exclude='.git/' --exclude='.venv/' --exclude='runs/' --exclude='.cache/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pdf' \
  --exclude='wcss/fleet_paths.env' --exclude='wcss/logs/' \
  --exclude='.env' --exclude='results/' --exclude='datasets/' \
  "$HERE/" "$WCSS_HOST:~/$WCSS_REMOTE/" || { echo "push failed"; exit 1; }

# Build the dataset on WCSS for BOTH index and rag (push excludes datasets/, so
# this (re)creates it to match the frozen index's docs).
echo ">> [1b] build dataset on WCSS (docs: $DOCS)"
ssh_r "cd ~/$WCSS_REMOTE && FINANCEBENCH_DIR=$FINANCEBENCH_DIR \
  python3 Scripts/preprocess/financebench_to_bookrag.py --docs $DOCS --out datasets/financebench.json" \
  >/dev/null || { echo "dataset build failed"; exit 1; }

EXPORTS="ALL,WORKDIR=\$HOME/$WCSS_REMOTE,NUM=$NUM,NSPLIT=$NSPLIT,ANSWER_MODEL=$ANSWER_MODEL"
# Answer-model matrix routing: gpt-* -> OpenAI cloud (only the LLM changes;
# Qwen embed+reranker stay, matching the frozen vdb); else the local Qwen fleet.
if [[ "$CMD" == "rag" ]]; then
  if [[ "$ANSWER_MODEL" == gpt-* ]]; then
    EXPORTS="$EXPORTS,CONFIG=config/financebench_wcss_gpt.yaml,CLOUD_LLM=1"
  else
    EXPORTS="$EXPORTS,CONFIG=config/financebench_wcss_local.yaml,CLOUD_LLM=0"
  fi
fi

echo ">> [2/3] submit $SLURM_FILE, then poll (resilient to login congestion)"
JID=$(ssh_r "cd ~/$WCSS_REMOTE && sbatch --parsable ${SBATCH_FLAGS[*]} --export=$EXPORTS $SLURM_FILE")
[[ -n "$JID" ]] || { echo "submit failed"; exit 1; }
echo ">> submitted job $JID — polling"
while true; do
  st=$(ssh_r "squeue -j $JID -h -o '%T' 2>/dev/null"); pc=$?
  [[ $pc -ne 0 ]] && { sleep 20; continue; }   # ssh failed — retry, don't exit
  [[ -z "$st" ]] && break                        # gone from queue -> finished
  sleep 30
done
STATE=$(ssh_r "sacct -j $JID -n -o State 2>/dev/null | head -1 | tr -d ' '")
echo ">> job $JID finished: ${STATE:-UNKNOWN}"
RC=0; [[ "$STATE" == COMPLETED* ]] || RC=1

echo ">> [3/3] pull artifacts (runs/ + results/ + logs)"
rsync_r "$WCSS_HOST:~/$WCSS_REMOTE/runs/" "$HERE/runs/" || true
if [[ "$CMD" == "rag" ]]; then
  rsync_r "$WCSS_HOST:~/$WCSS_REMOTE/results/" "$HERE/results/" || true
fi
rsync_r --include='bookrag-*.out' --include='bookrag-*.err' --exclude='*' \
  "$WCSS_HOST:~/$WCSS_REMOTE/" "$HERE/wcss/logs/" || true

echo ">> done (job $JID state=${STATE:-UNKNOWN} rc=$RC)"
exit $RC
exit $RC
