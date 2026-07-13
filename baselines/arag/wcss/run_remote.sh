#!/usr/bin/env bash
# LOCAL driver: rsync the A-RAG repo to WCSS, submit a SLURM job synchronously
# (sbatch --wait so DVC/loop blocks correctly), then pull results back.
#
#   wcss/run_remote.sh setup                 # one-time login-node setup
#   wcss/run_remote.sh index                 # build frozen 84-doc Qwen index
#   MODEL=gpt-4o-mini-2024-07-18 wcss/run_remote.sh answer
set -euo pipefail
ACTION="${1:?setup|index|answer}"
: "${WCSS:?set WCSS=<login>@ui.wcss.pl}"
# REMOTE is interpreted as $HOME-relative on the cluster (WORKDIR=$HOME/$REMOTE).
REMOTE="${REMOTE:-projects/arag}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo ">> rsync push -> $WCSS:$REMOTE"
# NEVER --delete data/ or results/: they are built ON WCSS and absent locally,
# so --delete would wipe the frozen index + predictions (BookRAG-learned trap).
rsync -az --delete \
  --exclude '.git' --exclude '.venv*' --exclude 'results' --exclude 'data' \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$HERE/" "$WCSS:$REMOTE/"

case "$ACTION" in
  setup)
    ssh "$WCSS" "cd $REMOTE && bash wcss/setup.sh"
    ;;
  index)
    ssh "$WCSS" "cd $REMOTE && export WORKDIR=\$HOME/$REMOTE && sbatch --wait wcss/index.slurm"
    ;;
  answer)
    MODEL="${MODEL:?set MODEL=<answer model id>}"
    ssh "$WCSS" "cd $REMOTE && export WORKDIR=\$HOME/$REMOTE MODEL=$MODEL && sbatch --wait --export=ALL,MODEL=$MODEL wcss/answer.slurm"
    echo ">> rsync pull results"
    rsync -az "$WCSS:$REMOTE/results/" "$HERE/results/"
    ;;
  *) echo "unknown action: $ACTION"; exit 2 ;;
esac
echo "REMOTE_DONE $ACTION"
