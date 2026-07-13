# A-RAG FinanceBench baseline

Reproduction harness for running [A-RAG](https://github.com/Ayanami0730/arag) (Du et al. 2026, arXiv:2602.03442) on FinanceBench as a baseline for ARAG-TOC.

## Setup

1. Clone upstream and pin the commit this harness was built against:

   ```bash
   git clone https://github.com/Ayanami0730/arag.git
   cd arag
   git checkout a44de6b2216bf6791979c4b6ac4ae106212fa1a6
   ```

2. Apply the upstream patch (adds run metadata to prediction records and token-median helpers in `src/arag/agent/base.py`):

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

Cluster access is configured via `WCSS` (ssh target, e.g. `your-login@ui.wcss.pl`) and `REMOTE` (remote checkout path). Secrets come from a gitignored `.env` (`OPENAI_API_KEY`); nothing is stored in the repo.

```bash
make setup                 # one-time: deps + tokenizer warmup on the login node
make index                 # build the frozen 84-doc Qwen3-Embedding index on H100
make answer                # 5-model answer matrix (sequential sbatch --wait)
make judge                 # judge locally in arag-toc -> metrics/arag_<model>.json
```

The judge is the same gpt-5-mini LLM judge used for the PageIndex and BookRAG baselines, so metrics are directly comparable.
