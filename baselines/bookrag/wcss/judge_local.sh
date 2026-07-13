#!/bin/bash
# Score BookRAG predictions with the SAME judge that grades PageIndex, which
# lives in the arag-toc repo. Copies predictions into arag-toc's results tree,
# runs its judge there, and writes metrics/bookrag_<model>.json next to
# metrics/pageindex_<model>.json. Run as the DVC `judge` stage.
#
# Env: ANSWER_MODEL, JUDGE_MODEL, PARALLEL, ARAG_TOC (defaults ../arag-toc).
set -euo pipefail

ANSWER_MODEL="${ANSWER_MODEL:-gpt-4o-mini}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
PARALLEL="${PARALLEL:-4}"
ARAG_TOC="${ARAG_TOC:-../arag-toc}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -d "$ARAG_TOC" ]] || { echo "ERROR: arag-toc not found at $ARAG_TOC (set ARAG_TOC=)"; exit 1; }
REL="results/financebench/bookrag/$ANSWER_MODEL"
mkdir -p "$ARAG_TOC/$REL"
cp "$HERE/$REL/predictions.json" "$ARAG_TOC/$REL/predictions.json"

( cd "$ARAG_TOC"
  set -a; . ./.env; set +a
  python -m eval.judge_predictions \
    "$REL/predictions.json" \
    --judge-model "$JUDGE_MODEL" --parallel "$PARALLEL" \
    --output "$REL/judged.json" \
    --metrics-out "metrics/bookrag_$ANSWER_MODEL.json" )

# Copy the metric back into the BookRAG repo so the DVC `judge` stage can track
# it (metrics/bookrag_<model>.json is the stage's metric output here too).
mkdir -p "$HERE/metrics"
cp "$ARAG_TOC/metrics/bookrag_$ANSWER_MODEL.json" "$HERE/metrics/bookrag_$ANSWER_MODEL.json"
echo "Judged -> $ARAG_TOC/$REL/judged.json  metrics -> $HERE/metrics/bookrag_$ANSWER_MODEL.json (+ arag-toc)"
