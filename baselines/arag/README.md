# A-RAG FinanceBench baseline

Reproduction harness for running [A-RAG](https://github.com/Ayanami0730/arag) (Du et al. 2026, arXiv:2602.03442) on FinanceBench as a baseline for ARAG-TOC.

## Prerequisites

Hard requirements before anything runs:

1. **A WCSS-like SLURM cluster with H100 GPUs.** ssh access set via `WCSS=<login>@host`, and a grant selected via `export SBATCH_ACCOUNT=...` (optionally `SBATCH_PARTITION=...`) before submitting.
2. **FinanceBench data on the cluster** at `$FINANCEBENCH_DIR/datasets/finance_bench/ground truth/financebench_open_source.jsonl` (default `$FINANCEBENCH_DIR=$HOME/finance_bench_root`).
3. **A BookRAG checkout at `$HOME/projects/BookRAG`** providing: the MinerU-parsed FinanceBench markdown at `runs/financebench/<uuid>/auto/*.md`, the ModelScope `Qwen/Qwen3-Embedding-0.6B` snapshot, and a `.env` holding `OPENAI_API_KEY` (used for the cloud answer models).
4. **A conda env named `bookrag`** under `$HOME/miniconda3` (has torch+cu126; `setup.sh` adds `sentence-transformers`).
5. **Locally**, a sibling `../arag-toc` checkout with its `.venv` and its own `.env` (`OPENAI_API_KEY`) for `make judge`, plus a sibling `../arag` upstream clone pinned to the commit below.

Two `.env` files are involved: **BookRAG's `.env` on the cluster** (used by the index/answer jobs) and **arag-toc's `.env` locally** (used by the judge). Neither is stored in this repo.

## Setup

1. Clone upstream and pin the commit this harness was built against:

   ```bash
   git clone https://github.com/Ayanami0730/arag.git
   cd arag
   git checkout a44de6b2216bf6791979c4b6ac4ae106212fa1a6
   ```

2. Apply the upstream patch (adds run metadata to prediction records and threads per-run input/output token accounting through the answer loop in `src/arag/agent/base.py`):

   ```bash
   git apply /path/to/baselines/arag/upstream.patch
   ```

3. Copy this directory's contents (`financebench/`, `wcss/`, `Makefile`) into the clone root.

## Layout

| Path | Purpose |
|------|---------|
| `financebench/financebench_to_arag.py` | Convert FinanceBench docs/questions into A-RAG's input format |
| `financebench/arag_patches.py` | Runtime patches (OpenAI-compatible endpoint auth, model wiring) |
| `financebench/financebench_run.py` | Answer-loop runner; model/endpoint via `ARAG_MODEL` / `ARAG_BASE_URL` / `ARAG_API_KEY` |
| `financebench/export_predictions.py` | Export predictions for the shared arag-toc judge |
| `wcss/` | SLURM scripts for the WCSS H100 cluster (index build, answer matrix, status probe) |
| `Makefile` | Gate-ladder targets: `setup` → `index` → `answer` → `judge` |

## Running

Cluster access is configured via `WCSS` (ssh target, e.g. `your-login@ui.wcss.pl`) and `REMOTE` (remote checkout path, interpreted as `$HOME`-relative). Secrets come from the gitignored `.env` files described above.

```bash
make setup                 # one-time: deps + tokenizer warmup on the login node
make index                 # build the frozen 84-doc Qwen3-Embedding index on H100
make answer                # 5-model answer matrix (sequential sbatch --wait)
make judge                 # judge locally in arag-toc -> metrics/arag_<model>.json
```

`make judge` uses arag-toc's shared LLM judge, pinned to the **same judge as the PageIndex and BookRAG baselines** so metrics are directly comparable. `judge_local.sh` passes no model flag, so the judge model is whatever arag-toc defaults to (`gpt-5.4-mini`); override it by exporting `LLM_JUDGE_MODEL_ID` before `make judge`.

## License / attribution

Upstream [`Ayanami0730/arag`](https://github.com/Ayanami0730/arag) (A-RAG, Du et al. 2026, arXiv:2602.03442) carries no license. `upstream.patch` is distributed as a minimal research-reproduction change set against pinned commit `a44de6b`, not a redistribution of upstream source. FinanceBench data (Patronus AI) is licensed CC BY-NC 4.0 and is downloaded by the user, not shipped here.
