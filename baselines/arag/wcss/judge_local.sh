#!/usr/bin/env bash
# LOCAL judge: score each model's A-RAG predictions with the arag-toc shared
# judge (gpt-5.4-mini, fixed) -> metrics/arag_<model>.json, next to
# metrics/pageindex_*.json and metrics/bookrag_*.json. Apples-to-apples.
set -euo pipefail
ARAG_TOC="${ARAG_TOC:-../arag-toc}"
ARAG="${ARAG:-../arag}"

# Resolve to absolute paths now: the script cd's into $ARAG_TOC, after which a
# relative $ARAG would point at the wrong place.
for var in ARAG_TOC ARAG; do
  dir="${!var}"
  [ -d "$dir" ] || { echo "$var directory not found: $dir" >&2; exit 1; }
  printf -v "$var" '%s' "$(cd "$dir" && pwd)"
done
MODELS="${MODELS:-gpt-5-mini-2025-08-07 gpt-5-2025-08-07 gpt-4.1-mini-2025-04-14 gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18}"
PARALLEL="${PARALLEL:-4}"

cd "$ARAG_TOC"
set -a; . ./.env; set +a
for m in $MODELS; do
  pred="$ARAG/results/financebench/arag/$m/predictions.json"
  if [ ! -f "$pred" ]; then echo "[skip] $m: no predictions.json"; continue; fi
  echo ">> judge $m"
  .venv/bin/python -m eval.judge_predictions "$pred" \
    --parallel "$PARALLEL" \
    --output "$ARAG/results/financebench/arag/$m/judged.json" \
    --metrics-out "metrics/arag_$m.json"
done
echo "JUDGE_DONE"
