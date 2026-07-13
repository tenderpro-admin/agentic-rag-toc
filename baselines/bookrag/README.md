# BookRAG FinanceBench baseline

Reproduction harness for running [BookRAG](https://github.com/sam234990/BookRAG) on FinanceBench as a baseline for ARAG-TOC.

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

3. Copy this directory's contents into the clone root (preserving paths: `config/`, `Scripts/`, `Eval/`, `wcss/`, `hpc/`, `datasets/`, `dvc.yaml`, `dvc.lock`, `params.yaml`, `Makefile`, `.dvc/`, `.dvcignore`).

## Layout

| Path | Purpose |
|------|---------|
| `Scripts/preprocess/financebench_to_bookrag.py` | Convert FinanceBench docs/questions into BookRAG's dataset format |
| `Scripts/cfg/financebench*.yaml` | Dataset configs (full 150-question set + Boeing smoke subset) |
| `config/financebench_*.yaml` | Model/provider configs (WCSS local Qwen fleet, OpenAI-endpoint variants) |
| `Eval/export_predictions.py` | Export predictions for the shared arag-toc judge |
| `wcss/` | SLURM scripts: MinerU indexing, vLLM model fleet, RAG answer runs |
| `hpc/watchdog.sh` | Job watchdog for long index builds |
| `dvc.yaml` / `dvc.lock` / `params.yaml` | Answer-model ablation matrix (answer@model → judge@model) |
| `datasets/financebench_boeing_smoke.json` | Small smoke-test subset |

## Pipeline

The BookIndex (MinerU parse → tree → graph → vdb) is built **once** on the WCSS H100 cluster with an all-local Qwen fleet and frozen; every matrix instance answers over the identical frozen index, so the only variable is the answering LLM.

```bash
make wcss-index DOCS=...      # one-off frozen index build on WCSS
uv run dvc repro              # all answer models + judge
uv run dvc repro judge@gpt-4o-2024-08-06   # one model end-to-end
uv run dvc metrics show       # compare metrics/bookrag_*
```

Cluster access via `WCSS_HOST` (e.g. `your-login@ui.wcss.pl`). Secrets come from a gitignored `.env` (`OPENAI_API_KEY`) only; nothing is stored in the repo. The judge is the same gpt-5-mini LLM judge used for the PageIndex and A-RAG baselines.
