# BookRAG × FinanceBench on WCSS (Lem GPU / H100)

Run the VRAM-heavy BookRAG indexing on the WCSS H100 cluster, then answer and
judge with the **same** metrics tracking as the PageIndex baseline in `arag-toc`.

```
 index  (WCSS H100)            answer (WCSS H100)         judge (local, arag-toc)
 ─────────────────────         ──────────────────        ───────────────────────
 dataset build (remote)        RAG over frozen index     copy predictions →
 MinerU PDF parse  (cuda)  →   reranker/embeds (cuda) →  eval.judge_predictions
 tree → graph → vdb            export_predictions.py     (gpt-5-mini, FIXED)
 → runs/financebench/<uuid>    → predictions.json        → metrics/bookrag_<m>.json
```

## Why this shape

The full upstream `config/gbc.yaml` fleet needs an 8-GPU node and 6 long-lived
vLLM/sglang servers (Qwen3-8B, Qwen2.5-VL, gme-Qwen2-VL, sglang MinerU, two
rerankers). We don't need that to *index*. `config/financebench_wcss.yaml` keeps
the LLM/VLM/embeddings on **OpenAI cloud** (gpt-4o-mini) and moves only the two
VRAM-bound pieces to **one H100**:

- **MinerU** PDF parse → `MINERU_DEVICE_MODE=cuda` (OOM'd on a 4 GB laptop GPU,
  slow on Mac MPS; on a 96 GB H100 it flies — 190-page 10-K layout in seconds).
- **Qwen3-Reranker-0.6B** → `device: cuda`, `backend: local` (no server).

Single GPU, single process, no server orchestration. Upgrading to the local-LLM
vLLM fleet is a future optimization, not required to index.

## Cluster facts (baked into the scripts)

| Thing | Value |
|---|---|
| Login | `ssh <your-login>@ui.wcss.pl` |
| GPU partition | `lem-gpu` (smoke: `lem-gpu-short`) |
| GPU request | `--gres=gpu:hopper:1` (H100, 96 GB) |
| Account | `hpc-tkajdanowicz-1763478893` |
| Checkout | `~/projects/BookRAG` |
| FinanceBench data | `~/finance_bench_root/datasets/finance_bench/` |
| conda env | `bookrag` (py3.12, torch 2.7.1+cu126) |
| Model cache | `~/projects/BookRAG/.cache/hf` (offline on compute nodes) |
| Secrets | `~/projects/BookRAG/.env` → `OPENAI_API_KEY` |

## One-time setup

```bash
make wcss-setup       # rsync repo + build conda env + pre-download all models
```

`wcss/setup_env.sh` runs on the **login node** (the only node with internet):
creates the `bookrag` env, `pip install -r requirements.txt` plus
`mineru[core]==2.1.11 ftfy dill`, pre-downloads the MinerU pipeline models and
the Qwen reranker into the persistent HF cache. Jobs then run with
`HF_HUB_OFFLINE=1` because GPU compute nodes are treated as offline.

## Run

```bash
# DVC (preferred — index -> answer -> judge, with metrics):
uv run dvc repro
SMOKE=1 DOCS=BOEING_2022_10K uv run dvc repro index   # single-doc smoke

# or via make:
make wcss-smoke                       # BOEING end to end
make wcss-index DOCS=all              # full 84-doc set
make wcss-answer MODEL=gpt-4o-mini
make wcss-judge  MODEL=gpt-4o-mini
```

`wcss/run_remote.sh` is **synchronous** (`sbatch --wait`) and returns the job's
exit code, so `dvc repro` blocks until the cluster job finishes and fails if it
fails. It pushes the checkout, builds the dataset on WCSS (so `doc_path` points
at the remote PDFs), submits the SLURM job, and rsyncs `runs/` + `results/` back.

## Gotchas handled (WCSS skill)

- **Absolute paths**: the upstream dataset cfg uses `../datasets/...`, which only
  resolves with `CWD=Scripts/`. `_common.sh` generates an absolute-path dataset
  cfg per job (paths are host-specific via `$WORKDIR`).
- **Cache isolation**: ephemeral caches (pip/triton/inductor) go to `$TMPDIR`;
  the model cache is persistent + pre-populated (compute nodes are offline).
- **Silent success**: `main.py` logs some failures yet exits 0 — `_common.sh`
  asserts the expected `runs/.../` artifacts exist and fails the job otherwise.

## Debugging

```bash
make wcss-status                                  # squeue
ssh <your-login>@ui.wcss.pl 'tail -f ~/projects/BookRAG/bookrag-index-<jobid>.err'
ssh <your-login>@ui.wcss.pl 'sacct -j <jobid> --format=JobID,State,ExitCode,Reason'
```

The `onnxruntime ... pthread_setaffinity_np failed` lines in `.err` are benign
(thread-affinity warnings from MinerU's OCR) — the job still runs on GPU.
