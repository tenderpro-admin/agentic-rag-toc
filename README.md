# ARAG TOC Benchmark Runner

This repository contains the benchmark workflow used for the ARAG-TOC paper's long-document question answering experiments on FinanceBench. It packages local TOC extraction, SQLite-backed indexing, agentic retrieval, and evaluation behind a single CLI.

## Requirements

- Python 3.10 or newer
- `uv`
- An OpenAI API key for answer generation and judging
- Network access

Retrieval embeddings default to `Qwen/Qwen3-Embedding-0.6B` through Sentence Transformers. The first run downloads that model into your Hugging Face cache.

## Setup

From a fresh checkout:

```bash
git clone <repo-url>
cd agentic-rag-toc
uv sync --dev
cp env_example.txt .env
uv run python scripts/fetch_bench_data.py
```

This downloads FinanceBench questions, document metadata, and PDFs into `datasets/finance_bench/`.

If you also want the optional PageIndex baseline, initialize the bundled submodule:

```bash
git submodule update --init --recursive
```

## Environment

`app_platform.config` auto-loads `.env` from the repo root. Start from `env_example.txt` and set values appropriate for your machine.

Important notes:

- `OPENAI_API_KEY` is required.
- `EMBEDDING_MODEL` defaults to `Qwen/Qwen3-Embedding-0.6B`; leave it unset unless you want a different embedding model.
- `OPENAI_REASONING_EFFORT` controls the reasoning setting for supported OpenAI models such as `gpt-5-mini`.
- `DATABASE_URL` is optional when using the default SQLite artifact path.
- PDF ingestion in this repo uses Docling.

The answer-generation path and evaluator both use LiteLLM's `openai/<model>` routing. If `LLM_JUDGE_MODEL_ID` is unset, the judge falls back to `OPENAI_JUDGE_MODEL` and then the built-in default.

## Quickstart

Check the CLI:

```bash
uv run python -m eval.qa --help
```

Run a one-question smoke test:

```bash
DOCLING_DEVICE=cpu \
uv run python -m eval.qa \
  --benchmark-source financebench \
  --limit 1
```

This creates or reuses a SQLite database under `.benchmark_artifacts/financebench/open_source/` and writes run artifacts under `results/financebench/open_source/`.

## `eval.qa` Arguments

The main entrypoint is `uv run python -m eval.qa`.

Core arguments:

- `--benchmark-source financebench`: run the FinanceBench benchmark data bundled for this repo.
- `--benchmark-config <name>`: select a benchmark config.
- `--limit <n>`: run only the first `n` test cases.
- `--sample-rate <n>`: sample every `n`th test case.
- `--parallel <n>`: set worker concurrency.

Filtering arguments:

- `--question <text>`: filter by question ID substring.
- `--source <text>`: filter by company or document identifier substring.
- `--subset <path>`: load question ID prefixes from a file, one per line.

Storage and output arguments:

- `--sqlite-db <path>`: override the SQLite database path for the run.
- `--reset-sqlite-db`: clear reusable SQLite benchmark tables before running.
- `--results-dir <path>`: override the directory used for run artifacts.
- `--commit <hash>`: include a git commit hash in the results metadata.

Mode arguments:

- `--predictions-only`: generate predictions without running the judge.
- `--index-only`: only index document chunks, without answering or judging.
- `--predictions-file <path>`: judge an existing `predictions_<timestamp>.json` artifact.
- `--resume <path>`: resume from a previous `predictions_*.json` or `qa_eval_*.json` file and skip completed cases.

## Common Uses

Judge an existing predictions artifact:

```bash
uv run python -m eval.qa \
  --predictions-file results/financebench/open_source/<run_timestamp>/predictions_<timestamp>.json
```

Run with a dedicated SQLite file:

```bash
DOCLING_DEVICE=cpu \
uv run python -m eval.qa \
  --benchmark-source financebench \
  --limit 1 \
  --sqlite-db .benchmark_artifacts/financebench/open_source/docling_toc_check.sqlite
```

Reuse a SQLite file cleanly between runs:

```bash
DOCLING_DEVICE=cpu \
uv run python -m eval.qa \
  --benchmark-source financebench \
  --limit 1 \
  --sqlite-db .benchmark_artifacts/financebench/open_source/docling_toc_check.sqlite \
  --reset-sqlite-db
```

Reproduce a larger paper-style run:

```bash
DOCLING_DEVICE=cpu \
uv run python -m eval.qa \
  --benchmark-source financebench \
  --parallel 10 \
  --sqlite-db .benchmark_artifacts/financebench/open_source/benchmark.sqlite \
  --results-dir results/financebench/open_source
```

Optional PageIndex baseline:

```bash
uv run python -m eval.pageindex_bench --parallel 10
```

The first PageIndex run builds a local workspace under `.benchmark_artifacts/pageindex/workspace`.

## External Baselines

Reproduction harnesses for the two external baselines evaluated in the paper live under `baselines/`:

- [`baselines/arag/`](baselines/arag/README.md) — A-RAG (Du et al. 2026) on FinanceBench
- [`baselines/bookrag/`](baselines/bookrag/README.md) — BookRAG on FinanceBench

Both are judged with the same LLM judge as the PageIndex baseline, so `metrics/*.json` are directly comparable across systems. See each README for prerequisites (both require a SLURM GPU cluster for indexing/answering).

## Output Locations

Defaults:

- SQLite DB: `.benchmark_artifacts/financebench/open_source/benchmark.sqlite`
- Results directory: `results/financebench/open_source/`

Fresh runs create a timestamped subdirectory inside the results directory.

Typical output files:

- Predictions: `results/financebench/open_source/<run_timestamp>/predictions_<timestamp>.json`
- Eval summary: `results/financebench/open_source/<run_timestamp>/qa_eval_<timestamp>.json`
- Agent log: `results/financebench/open_source/<run_timestamp>/logs/agentic_rag_<timestamp>.json`

## Troubleshooting

### Missing OpenAI credentials

If a run fails with an authentication error, make sure `OPENAI_API_KEY` is set in `.env` or exported in your shell.

### Docling on macOS

If Docling hits device issues on macOS, force CPU:

```bash
DOCLING_DEVICE=cpu uv run python -m eval.qa --benchmark-source financebench --limit 1
```

### Resetting a reused SQLite file

If you reuse a benchmark SQLite file, prefer the built-in reset flag:

```bash
uv run python -m eval.qa \
  --benchmark-source financebench \
  --sqlite-db .benchmark_artifacts/financebench/open_source/docling_toc_check.sqlite \
  --reset-sqlite-db
```

The CLI clears `haystack_documents`, `haystack_documents_fts`, and `document_toc` when those tables exist.
