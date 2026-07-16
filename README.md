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

## Baselines

PageIndex, BookRAG, and A-RAG are reproduced locally and evaluated with the same FinanceBench data and judge. Complete [Setup](#setup) first so `.env` and `datasets/finance_bench/` are available.

| Baseline | Index and retrieval | Setup |
| --- | --- | --- |
| [PageIndex](#pageindex) | LLM-generated hierarchical tree; no embeddings | Bundled Git submodule |
| [BookRAG](#bookrag) | MinerU, graph construction, local Qwen embeddings and reranking | Repository bootstrap script |
| [A-RAG](#a-rag) | Local Qwen embeddings over BookRAG's MinerU text | Repository bootstrap script |

BookRAG must run before A-RAG because A-RAG consumes its parsed MinerU markdown. Start each baseline with its one-case smoke command before launching a full run.

### PageIndex

Initialize the bundled [PageIndex](https://github.com/VectifyAI/PageIndex) submodule, then run one case:

```bash
git submodule update --init --recursive
uv run python -m eval.pageindex_bench --limit 1
```

Run the full prediction set with:

```bash
uv run python -m eval.pageindex_bench --parallel 10
```

PageIndex writes timestamped predictions to `results/financebench/pageindex/` and keeps its local workspace under `.benchmark_artifacts/pageindex/workspace/`.

### BookRAG

The bootstrap script clones the pinned [BookRAG](https://github.com/sam234990/BookRAG) revision, installs the [`baselines/bookrag/`](baselines/bookrag/README.md) overlay, links `.env`, creates its environment, and validates the FinanceBench data:

```bash
./scripts/setup_bookrag.sh
make -C BookRAG smoke FINANCEBENCH_DIR="$(pwd)"
```

The smoke test uses one question from the shortest eligible document, the four-page `FOOTLOCKER_2022_8K_dated-2022-05-20`. Its prediction is written to `BookRAG/results/financebench/bookrag-smoke/gpt-4o-mini/predictions.json`.

After the smoke test succeeds, run the full baseline:

```bash
make -C BookRAG index DOCS=all FINANCEBENCH_DIR="$(pwd)"
make -C BookRAG answer MODEL=gpt-4o-mini
make -C BookRAG judge MODEL=gpt-4o-mini ARAG_TOC="$(pwd)"
```

BookRAG uses local `Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-0.6B`; OpenAI is used for graph and answer generation. Graph construction can use substantially more tokens than answering. Set `MINERU_DEVICE=cpu`, `mps`, or `cuda` to override automatic MinerU device selection.

The setup script is safe to rerun at the pinned revision. It also supports `--prepare-only` and a custom destination; see the [BookRAG harness README](baselines/bookrag/README.md) for manual setup and configuration details.

### A-RAG

After BookRAG has produced MinerU markdown, clone and configure the pinned [A-RAG](https://github.com/Ayanami0730/arag) revision with its [`baselines/arag/`](baselines/arag/README.md) overlay:

```bash
./scripts/setup_arag.sh
make -C arag smoke \
  FINANCEBENCH_DIR="$(pwd)" \
  BOOKRAG_RUNS_DIR="$(pwd)/BookRAG/runs/financebench_qwen_smoke"
```

The smoke prediction is written to `arag/results/financebench/arag-smoke/gpt-4o-mini/predictions.json`.

For the full baseline, use BookRAG's full run directory:

```bash
make -C arag index DOCS=all \
  FINANCEBENCH_DIR="$(pwd)" \
  BOOKRAG_RUNS_DIR="$(pwd)/BookRAG/runs/financebench"
make -C arag answer MODEL=gpt-4o-mini
make -C arag judge MODEL=gpt-4o-mini ARAG_TOC="$(pwd)"
```

`DEVICE=auto` selects CUDA, then MPS, then CPU. A-RAG writes resumable answer rows and exports validated predictions to `arag/results/financebench/arag/<model>/predictions.json`. The setup script is idempotent and supports `--prepare-only`; see the [A-RAG harness README](baselines/arag/README.md) for manual setup and endpoint configuration.

### Shared Evaluation

The main ARAG-TOC runner and all three baselines produce prediction artifacts accepted by `postprocessing.evaluator`. The predictions file is the only positional input that changes between methods:

```bash
uv run python -m postprocessing.evaluator \
  results/financebench/pageindex/predictions_<timestamp>.json \
  --parallel 4 \
  --judge-model gpt-5.4-mini
```

All paths use `postprocessing.evaluator`. Their `qa_eval_<timestamp>.json` artifacts share the main benchmark's result schema. Override the Make targets with `JUDGE_MODEL=<model>` and `PARALLEL=<n>` when needed.

## Output Locations

Prediction locations:

| Method | Standard predictions | Smoke predictions |
| --- | --- | --- |
| ARAG-TOC | `results/financebench/open_source/<run_timestamp>/predictions_<timestamp>.json` | Same path pattern with a one-case run timestamp |
| PageIndex | `results/financebench/pageindex/predictions_<timestamp>.json` | Same path with `--limit 1` |
| BookRAG | `BookRAG/results/financebench/bookrag/<model>/predictions.json` | `BookRAG/results/financebench/bookrag-smoke/<model>/predictions.json` |
| A-RAG | `arag/results/financebench/arag/<model>/predictions.json` | `arag/results/financebench/arag-smoke/<model>/predictions.json` |

ARAG-TOC defaults:

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
