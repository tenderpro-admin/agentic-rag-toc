#!/usr/bin/env bash
# LOCAL judge: score each model's A-RAG predictions with the arag-toc shared
# judge (gpt-5.4-mini, fixed) -> metrics/arag_<model>.json, next to
# metrics/pageindex_*.json and metrics/bookrag_*.json. Apples-to-apples.
set -euo pipefail
ARAG_TOC="${ARAG_TOC:-/home/bukareszt/Downloads/aragtoc/arag-toc}"
ARAG="${ARAG:-/home/bukareszt/Downloads/aragtoc/arag}"
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
