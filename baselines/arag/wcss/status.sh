#!/usr/bin/env bash
# A-RAG status probe. Prints ONE JSON line for the loop's gate ladder.
# Run on WCSS (reads ARAG_DATA + results). Counts frozen indexes, per-model
# prediction lines, exported predictions.json, and running SLURM jobs.
set -euo pipefail
cd "${WORKDIR:-$HOME/projects/arag}"
BOOKRAG="${BOOKRAG:-$HOME/projects/BookRAG}"
ARAG_DATA="${ARAG_DATA:-$WORKDIR/data/financebench}"
RESULTS_DIR="${RESULTS_DIR:-$WORKDIR/results/financebench/arag}"
RUNS="${RUNS:-$BOOKRAG/runs/financebench}"

MODELS="${MODELS:-gpt-5-mini-2025-08-07 gpt-5-2025-08-07 gpt-4.1-mini-2025-04-14 gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18}"

docs_total=$(find "$RUNS" -maxdepth 2 -name '*.md' -path '*/auto/*' 2>/dev/null | wc -l)
indexes=$(find "$ARAG_DATA" -name sentence_index.pkl 2>/dev/null | wc -l)
metrics=0
preds_json=""
for m in $MODELS; do
  n=0
  [ -f "$RESULTS_DIR/$m/predictions.jsonl" ] && n=$(grep -c . "$RESULTS_DIR/$m/predictions.jsonl" 2>/dev/null || echo 0)
  preds_json="$preds_json\"$m\":$n,"
  [ -f "$WORKDIR/metrics/arag_$m.json" ] && metrics=$((metrics+1))
done
preds_json="{${preds_json%,}}"
jobs=$(squeue -u "$USER" -h -o '%i' -n arag-index,arag-answer 2>/dev/null | tr '\n' ',' | sed 's/,$//')

printf '{"indexes":%s,"docs_total":%s,"preds":%s,"metrics":%s,"jobs_running":"%s"}\n' \
  "$indexes" "$docs_total" "$preds_json" "$metrics" "$jobs"
