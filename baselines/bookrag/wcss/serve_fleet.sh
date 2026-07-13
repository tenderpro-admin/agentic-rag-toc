#!/bin/bash
# Launch BookRAG's OFFICIAL local Qwen fleet on ONE WCSS node (4x H100), wait
# until every server is healthy, run main.py (index or rag), tear the fleet down.
#
# The 4 vLLM servers run inside the cluster's vLLM apptainer container (no conda
# dependency conflict with mineru/transformers). MinerU parses with its GPU
# 'pipeline' backend and the gme vdb embedder loads in-process — both inside
# main.py on GPU3. main.py sees all 4 GPUs; the servers are masked to one each.
#
#   GPU0 vLLM(container) Qwen3-8B-AWQ           :8003/v1   (LLM)
#   GPU1 vLLM(container) Qwen2.5-VL-7B-Instruct :8000/v1   (VLM)
#   GPU2 vLLM(container) Qwen3-Embedding-0.6B   :8007/v1   (embed)
#   GPU2 vLLM(container) Qwen3-Reranker-4B      :8011      (rerank -> /rerank)
#   GPU3 main.py: MinerU pipeline (cuda:3) + gme-Qwen2-VL-2B (cuda:3)
set -uo pipefail

: "${WORKDIR:?WORKDIR must be set}"
export CONFIG="${CONFIG:-config/financebench_wcss_local.yaml}"
export COMMAND="${COMMAND:-index}"
export STAGE="${STAGE:-all}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

source "$WORKDIR/wcss/_common.sh"
# Keep MinerU + gme off the GPUs the vLLM servers own (0,1,2).
export MINERU_DEVICE_MODE="cuda:3"

# Local ModelScope paths for every fleet model (written by setup_fleet.sh). The
# HF route from WCSS is throttled, so we serve straight from these cached dirs.
PATHS_ENV="$WORKDIR/wcss/fleet_paths.env"
[[ -f "$PATHS_ENV" ]] || { echo "missing $PATHS_ENV — run wcss/setup_fleet.sh first"; exit 1; }
# shellcheck disable=SC1090
source "$PATHS_ENV"
: "${LLM_PATH:?}" "${EMBED_PATH:?}" "${RERANK_PATH:?}"
# VLM and gme are OPTIONAL. The big models (Qwen2.5-VL 16G, gme 5G, Reranker-4B 9G)
# are flaky to fetch over the WCSS↔ModelScope link; FinanceBench 10-Ks are
# text-heavy, so we run an all-cached small fleet (Qwen3-8B-AWQ + Qwen3-Embedding
# + Qwen3-Reranker-0.6B). GME_PATH absent -> the config keeps its server-based
# text embedder and no multimodal path; VLM_PATH absent -> image descriptions off.
VLM_ENABLED=0
[[ -n "${VLM_PATH:-}" && -d "${VLM_PATH:-/nonexistent}" ]] && VLM_ENABLED=1
GME_ENABLED=0
[[ -n "${GME_PATH:-}" && -d "${GME_PATH:-/nonexistent}" ]] && GME_ENABLED=1

# Patch per-job config: substitute the local gme path only if gme is present;
# disable VLM image-description features when VLM is absent.
# When answering via a cloud LLM (CLOUD_LLM=1), set llm.model_name = ANSWER_MODEL
# so a single gpt config serves the whole answer-model matrix (gpt-4.1-mini,
# gpt-4o, ...). Empty unless CLOUD_LLM=1.
LLM_OVERRIDE=""
[[ "${CLOUD_LLM:-0}" == "1" ]] && LLM_OVERRIDE="${ANSWER_MODEL:-}"

PATCHED_CONFIG="$TMPDIR/$(basename "$CONFIG" .yaml)_patched.yaml"
python - "$WORKDIR/$CONFIG" "$PATCHED_CONFIG" "${GME_PATH:-}" "$VLM_ENABLED" "$GME_ENABLED" "$LLM_OVERRIDE" <<'PY'
import sys, yaml
src, dst, gme, vlm_enabled, gme_enabled, llm_override = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1", sys.argv[5] == "1", sys.argv[6])
c = yaml.safe_load(open(src))
if gme_enabled:
    c.setdefault("vdb", {}).setdefault("embedding_config", {})["model_name"] = gme
    c.setdefault("rag", {}).setdefault("mm_reranker_config", {})["model_name"] = gme
if not vlm_enabled:
    c.setdefault("graph", {})["image_description_force"] = False
    c.setdefault("tree", {})["use_vlm"] = False
    c.setdefault("vdb", {})["mm_embedding"] = False
if llm_override:  # cloud answer-model ablation: only the LLM changes
    import os
    llm = c.setdefault("llm", {})
    llm["model_name"] = llm_override
    # Upstream BookRAG passes api_key verbatim (the `env`->OPENAI_API_KEY provider
    # patch lives in our stash). Resolve it here so `api_key: env` isn't sent as a
    # literal "env" -> 401. The patched config stays in $TMPDIR (tmpfs), never repo.
    if str(llm.get("api_key", "")).strip() in {"env", "EMPTY", ""}:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("CLOUD_LLM=1 but OPENAI_API_KEY not set")
        llm["api_key"] = key
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
print("patched | vlm:", vlm_enabled, "| gme:", gme_enabled, "| llm:", llm_override or "(config)")
PY
CONFIG="$PATCHED_CONFIG"
export CONFIG

VLLM_SIF="${VLLM_SIF:-/lustre/software-data/container-images/vllm-gpu-v0.9.0.1.sif}"
[[ -f "$VLLM_SIF" ]] || { echo "vLLM container not found: $VLLM_SIF"; exit 1; }
command -v apptainer >/dev/null 2>&1 || source /usr/local/sbin/modules.sh 2>/dev/null || true

LOGDIR="$WORKDIR/wcss/logs/fleet-${SLURM_JOB_ID:-local}"
mkdir -p "$LOGDIR"
PIDS=()

cleanup() {
  echo ">> tearing down fleet"
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_health() {  # name url timeout_s
  local name="$1" url="$2" t="${3:-900}" i=0
  echo -n ">> waiting for $name ($url) "
  while (( i < t )); do
    if curl -sf "$url" >/dev/null 2>&1; then echo " UP (${i}s)"; return 0; fi
    sleep 5; i=$((i+5)); echo -n "."
  done
  echo " TIMEOUT after ${t}s"; tail -n 30 "$LOGDIR/$name.log" 2>/dev/null; return 1
}

serve() {  # gpu name port "vllm serve args..."
  local gpu="$1" name="$2" port="$3" args="$4"
  # Per-server WRITABLE cache inside the container. The vLLM .sif defaults its
  # caches to a read-only path (/mnt/lscratch) -> "OSError: Read-only file
  # system". Redirect HOME + every framework cache into the bound $TMPDIR.
  local c="$TMPDIR/vllm-$name"
  mkdir -p "$c/cache" "$c/triton" "$c/vllm" "$c/outlines" "$c/numba"
  echo ">> launch vLLM $name on GPU$gpu :$port"
  # If .cache is a symlink (e.g. -> /lustre/...), apptainer does not make the
  # symlink TARGET visible inside the container, so the bound .cache path is a
  # dangling symlink and vLLM cannot find the model dir. Bind the resolved
  # target at its own absolute path so the symlink resolves inside the container.
  local creal; creal="$(readlink -f "$WORKDIR/.cache" 2>/dev/null)"
  local cache_bind=""
  [ -n "$creal" ] && [ "$creal" != "$WORKDIR/.cache" ] && cache_bind="--bind $creal:$creal"
  # shellcheck disable=SC2086
  apptainer exec --nv \
    --bind "$WORKDIR/.cache:$WORKDIR/.cache" --bind "$TMPDIR:$TMPDIR" $cache_bind \
    --env CUDA_VISIBLE_DEVICES="$gpu" --env PYTHONUNBUFFERED=1 \
    --env HOME="$c" --env XDG_CACHE_HOME="$c/cache" \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env VLLM_CACHE_ROOT="$c/vllm" --env TRITON_CACHE_DIR="$c/triton" \
    --env OUTLINES_CACHE_DIR="$c/outlines" --env NUMBA_CACHE_DIR="$c/numba" \
    "$VLLM_SIF" \
    vllm serve $args --host 127.0.0.1 --port "$port" \
    > "$LOGDIR/$name.log" 2>&1 &
  PIDS+=($!)
}

# Local Qwen LLM server. Skipped when CLOUD_LLM=1 (answer/reasoning LLM is an
# OpenAI model in the config — see financebench_wcss_gpt.yaml). Embed + reranker
# stay identical so retrieval matches the frozen Qwen-built vdb.
if [[ "${CLOUD_LLM:-0}" == "1" ]]; then
  echo ">> CLOUD_LLM=1 — skipping local Qwen LLM server (answering via OpenAI cloud)"
else
  # Launch + health-gate the LLM FIRST and ALONE. The AWQ 8B is the heavy one; a
  # previous run hung it silently (0-byte log) while the small servers loaded —
  # likely CUDA-graph capture contention. --enforce-eager skips graph capture
  # (much faster, more robust startup); a shorter max-model-len trims KV profiling.
  serve 0 llm 8003 "$LLM_PATH --served-model-name Qwen/Qwen3-8B-AWQ --gpu-memory-utilization 0.85 --max-model-len 8192 --enforce-eager"
  wait_health llm "http://127.0.0.1:8003/health" 1800 || exit 1
fi

# Reranker is NOT served by vLLM: Qwen3-Reranker is a CausalLM scoring via yes/no
# token logits, and vLLM --task score (/rerank) 500s on it. BookRAG loads it
# in-process (backend: local, device cuda:3) — proven path. So only embed here.
serve 2 embed  8007 "$EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-0.6B --task embed --gpu-memory-utilization 0.18 --enforce-eager"
if [[ "$VLM_ENABLED" == "1" ]]; then
  serve 1 vlm 8000 "$VLM_PATH --served-model-name Qwen/Qwen2.5-VL-7B-Instruct --gpu-memory-utilization 0.65 --max-model-len 16384 --limit-mm-per-prompt image=4 --enforce-eager"
else
  echo ">> VLM disabled (no VLM_PATH) — running text-only, image descriptions off"
fi

wait_health embed  "http://127.0.0.1:8007/health" 900 || exit 1
[[ "$VLM_ENABLED" == "1" ]] && { wait_health vlm "http://127.0.0.1:8000/health" 1200 || exit 1; }
echo ">> fleet healthy — starting BookRAG $COMMAND"

# Force a FRESH answer per model: BookRAG loads existing final_results.json when
# present (rag_force_reprocess doesn't re-answer), so an answer-model ablation
# would reuse the previous model's outputs. Delete only the RAG output dir
# (eval_*/) — the frozen index (tree.pkl, Tree_vdb, kg_vdb in the run root) stays.
if [[ "$COMMAND" == "rag" ]]; then
  echo ">> clearing prior RAG outputs (eval_*/) so $ANSWER_MODEL re-answers fresh"
  rm -rf "$WORKDIR/runs/financebench"/*/eval_* 2>/dev/null || true
fi

bookrag_run
rc=$?

if [[ "$COMMAND" == "rag" && $rc -eq 0 ]]; then
  ANSWER_MODEL="${ANSWER_MODEL:-Qwen3-8B-AWQ}"
  PRED_OUT="${PRED_OUT:-results/financebench/bookrag/$ANSWER_MODEL/predictions.json}"
  echo ">> export_predictions -> $PRED_OUT"
  python -m Eval.export_predictions \
    --runs-dir "${RUNS_DIR:-runs/financebench}" --model "$ANSWER_MODEL" --out "$PRED_OUT"
  rc=$?
fi

echo "FLEET_RUN_DONE rc=$rc at $(date)"
exit $rc