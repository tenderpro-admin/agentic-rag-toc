# BookRAG × FinanceBench on WCSS (Lem GPU / H100)

Build the VRAM-heavy BookRAG index on the WCSS H100 cluster, run the answer-model
ablation over that frozen index, then judge locally with the **same** judge as the
PageIndex baseline in `arag-toc`.

```
 index  (WCSS H100, once)       answer (WCSS H100, per model)    judge (local, arag-toc)
 ─────────────────────────      ─────────────────────────────    ───────────────────────
 dataset build (remote)         RAG over the FROZEN index         copy predictions →
 MinerU PDF parse  (cuda)   →   local Qwen fleet OR cloud LLM  →  eval.judge_predictions
 tree → graph → vdb             export_predictions.py             (gpt-5.4-mini, FIXED)
 (local Qwen fleet)             → predictions.json                → metrics/bookrag_<m>.json
 → runs/financebench/<uuid>
```

## Why this shape

The BookIndex (MinerU parse → tree → graph → vdb) is expensive, so it is built
**once** with BookRAG's official all-local Qwen fleet on one 4× H100 node and then
**frozen**. Every ablation instance answers over that identical frozen index, so
the only variable is the answering LLM — an honest answer-model ablation. The
answer LLM is either the local `Qwen3-8B-AWQ` (served by `wcss/serve_fleet.sh`) or
a cloud OpenAI model (`CLOUD_LLM=1`, `config/financebench_wcss_gpt.yaml`); the
retrieval stack (Qwen embed + in-process Qwen reranker) stays fixed so it matches
the frozen vdb.

(`config/financebench_wcss.yaml` / `financebench_gpt4omini.yaml` keep a simpler
single-GPU, cloud-LLM index path for smoke/debug; the ablation itself uses the
local-fleet configs above.)

## Cluster facts (baked into the scripts)

| Thing | Value |
|---|---|
| Login | `ssh <your-login>@ui.wcss.pl` (set `WCSS_HOST`) |
| GPU partition | `lem-gpu` (smoke: `lem-gpu-short`) |
| GPU request | `--gres=gpu:hopper:4` (fleet index/answer); `:1` for the simple cloud path |
| Account | `<your-grant-account>` — `export SBATCH_ACCOUNT=...` before submitting; `sbatch` reads it from the environment (optionally `SBATCH_PARTITION`) |
| Checkout | `~/projects/BookRAG` |
| FinanceBench data | `~/finance_bench_root/datasets/finance_bench/` |
| conda env | `bookrag` (py3.12, torch 2.7.1+cu126) |
| Model cache | `~/projects/BookRAG/.cache/hf` (offline on compute nodes) |
| Fleet models | pre-downloaded by `wcss/setup_fleet.sh` → `wcss/fleet_paths.env` |
| Secrets | `~/projects/BookRAG/.env` → `OPENAI_API_KEY` (cloud LLM + judge only) |

## One-time setup

```bash
make wcss-setup                 # rsync repo + build conda env + pre-download models
ssh <your-login>@ui.wcss.pl 'cd ~/projects/BookRAG && bash wcss/setup_fleet.sh'
```

`wcss/setup_env.sh` runs on the **login node** (the only node with internet):
creates the `bookrag` env, installs deps, and pre-downloads the MinerU pipeline
models and the in-process Qwen reranker into the persistent HF cache.
`wcss/setup_fleet.sh` pre-downloads the vLLM fleet (Qwen3-8B-AWQ, embedder,
optional VLM/gme) from ModelScope and writes `wcss/fleet_paths.env`
(`serve_fleet.sh` hard-fails without it). Jobs then run with `HF_HUB_OFFLINE=1`
because GPU compute nodes are treated as offline.

## Run

```bash
export SBATCH_ACCOUNT=<your-grant-account>

# DVC (preferred — answer-model matrix over the frozen index -> judge, with metrics):
uv run dvc repro

# single-doc smoke via make:
make wcss-index DOCS=BOEING_2022_10K SMOKE=1
make wcss-answer MODEL=gpt-4o-mini DOCS=BOEING_2022_10K
make wcss-judge  MODEL=gpt-4o-mini

# full 84-doc set:
make wcss-index DOCS=all
```

`wcss/run_remote.sh` submits the job with `sbatch --parsable`, then **polls**
(`squeue`, falling back to `sacct` when the job leaves the queue) until it
finishes and exits with the job's final state — so `dvc repro` blocks until the
cluster job completes and fails if it fails. Polling (not `sbatch --wait`) keeps a
single congested login-node ssh from wedging the run. It pushes the checkout,
builds the dataset on WCSS (so `doc_path` points at the remote PDFs), submits, and
rsyncs `runs/` + `results/` back.

`dvc.lock` is **not committed**: it is produced by `dvc repro`, which requires the
private cluster; the committed state is the pipeline definition only.

## Gotchas handled (WCSS skill)

- **Absolute paths**: `_common.sh` generates an absolute-path dataset cfg per job
  (paths are host-specific via `$WORKDIR`).
- **Cache isolation**: ephemeral caches (pip/triton/inductor) go to `$TMPDIR`; the
  model cache is persistent + pre-populated (compute nodes are offline).
- **Silent success**: `main.py` logs some failures yet exits 0 — `_common.sh`
  asserts the expected `runs/.../` artifacts exist and catches a fresh
  `financebench-*_error*.txt` (index or rag), failing the job otherwise.

## Debugging

```bash
make wcss-status                                  # squeue
ssh <your-login>@ui.wcss.pl 'tail -f ~/projects/BookRAG/bookrag-index-<jobid>.err'
ssh <your-login>@ui.wcss.pl 'sacct -j <jobid> --format=JobID,State,ExitCode,Reason'
```

The `onnxruntime ... pthread_setaffinity_np failed` lines in `.err` are benign
(thread-affinity warnings from MinerU's OCR) — the job still runs on GPU.
