# BookRAG FinanceBench baseline

Reproduction harness for running [BookRAG](https://github.com/sam234990/BookRAG) on FinanceBench as a baseline for ARAG-TOC.

## Prerequisites

- **Tooling (local):** `uv` and `dvc` installed.
- **Cluster access:** ssh to the WCSS login node (`export WCSS_HOST=<your-login>@ui.wcss.pl`) and a grant to charge jobs to (`export SBATCH_ACCOUNT=<your-grant-account>`; `sbatch` reads it from the environment — no account is baked into the scripts).
- **Sibling `arag-toc` checkout** at `../arag-toc` (relative to this repo) with its `.venv` and a `.env` holding `OPENAI_API_KEY`. Judging runs there so BookRAG and PageIndex are scored by the identical judge.
- **FinanceBench data** laid out as `<root>/datasets/finance_bench/{ground truth,pdfs}`. Locally the converter defaults to the sibling `arag-toc` (which fetches it); on the cluster it lives at `~/finance_bench_root/` (set `FINANCEBENCH_DIR`). The data is CC BY-NC and is never committed here.
- **Cluster fleet:** run `wcss/setup_fleet.sh` once on the login node to pre-download the Qwen fleet and create `wcss/fleet_paths.env` — `wcss/serve_fleet.sh` hard-fails without it.
- **Upstream clone (cluster):** `sam234990/BookRAG` at pinned commit `113298f` with `upstream.patch` applied, placed at `$HOME/projects/BookRAG` (see Setup).

## Setup

1. Clone upstream and pin the commit this harness was built against:

   ```bash
   git clone https://github.com/sam234990/BookRAG.git
   cd BookRAG
   git checkout 113298f919c701d07807ceccb96ed3c18d348117
   ```

2. Apply the upstream patch (env-based API-key resolution in the LLM/rerank providers, CPU-reranker fallback, table-parsing and answer-loop fixes):

   ```bash
   git apply /path/to/baselines/bookrag/upstream.patch
   ```

3. Copy this directory's contents into the clone root (preserving paths: `config/`, `Scripts/`, `Eval/`, `wcss/`, `hpc/`, `dvc.yaml`, `params.yaml`, `Makefile`, `.dvc/`, `.dvcignore`, `.gitignore`). `datasets/`, `runs/`, `results/`, `metrics/` and `dvc.lock` are generated at run time (gitignored).

## Layout

| Path | Purpose |
|------|---------|
| `Scripts/preprocess/financebench_to_bookrag.py` | Convert FinanceBench docs/questions into BookRAG's dataset format |
| `Scripts/cfg/financebench*.yaml` | Dataset configs (full 150-question set + Boeing smoke subset) |
| `config/financebench_*.yaml` | Model/provider configs (WCSS local Qwen fleet, OpenAI-endpoint variants) |
| `Eval/export_predictions.py` | Export predictions for the shared arag-toc judge |
| `wcss/` | SLURM scripts: MinerU indexing, vLLM model fleet, RAG answer runs |
| `hpc/watchdog.sh` | Login-node zombie-process reaper for long index builds |
| `dvc.yaml` / `params.yaml` | Answer-model ablation matrix (answer@model → judge@model) |

## Pipeline

The BookIndex (MinerU parse → tree → graph → vdb) is built **once** on the WCSS H100 cluster with an all-local Qwen fleet and frozen; every matrix instance answers over the identical frozen index, so the only variable is the answering LLM.

```bash
make wcss-index DOCS=...      # one-off frozen index build on WCSS
uv run dvc repro              # all answer models + judge
uv run dvc repro judge@gpt-4o-2024-08-06   # one model end-to-end
uv run dvc metrics show       # compare metrics/bookrag_*
```

Cluster access via `WCSS_HOST` (e.g. `your-login@ui.wcss.pl`). Secrets come from a gitignored `.env` (`OPENAI_API_KEY`) only; nothing is stored in the repo. The judge is the same `gpt-5.4-mini` LLM judge used for the PageIndex and A-RAG baselines (override via `LLM_JUDGE_MODEL_ID`). `dvc.lock` is **not committed**: it is produced by `dvc repro` (which requires the private cluster); the committed state is the pipeline definition only.

See `wcss/README.md` for the cluster run details.

## License / attribution

Upstream `sam234990/BookRAG` ships **no license**; `upstream.patch` is a minimal research-reproduction change set against pinned commit `113298f`, not a redistribution of upstream. FinanceBench (Patronus AI) is **CC BY-NC 4.0**; the data is downloaded by the user (via arag-toc's fetch script) and is never committed to this repo.
