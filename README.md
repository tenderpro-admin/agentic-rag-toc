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

The BookRAG and A-RAG reproduction harnesses run locally on macOS or Linux. They support CUDA, Apple MPS, and CPU; no SLURM cluster is required. Both use `Qwen/Qwen3-Embedding-0.6B` locally and use the same evaluator as this repository.

Run BookRAG first. A-RAG does not parse PDFs itself and consumes the MinerU markdown produced by BookRAG, ensuring both baselines see the same document text.

### Shared baseline prerequisites

Complete the main repository setup and download FinanceBench before setting up either baseline:

```bash
uv sync --dev
cp env_example.txt .env
# Set OPENAI_API_KEY in .env.
uv run python scripts/fetch_bench_data.py
```

The resulting layout must contain:

```text
agentic-rag-toc/
|-- .env
|-- datasets/finance_bench/
|   |-- ground truth/
|   `-- pdfs/
|-- BookRAG/
`-- arag/
```

The first baseline run downloads local embedding, reranking, and PDF-processing models. BookRAG graph construction also makes OpenAI requests and can use substantially more tokens than answer generation, so start with the smoke workflow.

### BookRAG

The setup script clones the pinned [BookRAG](https://github.com/sam234990/BookRAG) commit, installs the local harness from [`baselines/bookrag/`](baselines/bookrag/README.md), links the root `.env`, creates the Python environment, and runs the dependency/data check:

```bash
export ARAG_TOC="$(pwd)"
export FINANCEBENCH_DIR="$ARAG_TOC"
./scripts/setup_bookrag.sh
cd BookRAG
```

The script is safe to rerun when `BookRAG/` is already at the expected commit. It refuses to switch an existing checkout at another commit. Use `./scripts/setup_bookrag.sh --prepare-only` to clone and install the overlay without creating the environment, or pass a destination as the final argument to install outside the repository. The equivalent manual process is documented in the baseline README.

Run the shortest available FinanceBench case first:

```bash
make smoke FINANCEBENCH_DIR="$FINANCEBENCH_DIR"
```

The smoke test uses the four-page `FOOTLOCKER_2022_8K_dated-2022-05-20` document and one question. It writes:

```text
BookRAG/runs/financebench_qwen_smoke/
BookRAG/results/financebench/bookrag-smoke/gpt-4o-mini/predictions.json
```

Run the full BookRAG baseline after the smoke test succeeds:

```bash
make index DOCS=all FINANCEBENCH_DIR="$FINANCEBENCH_DIR"
make answer MODEL=gpt-4o-mini
make judge MODEL=gpt-4o-mini ARAG_TOC="$ARAG_TOC"
```

`make index` builds the FinanceBench dataset, parses PDFs with local MinerU, constructs the graph, and indexes with local Qwen embeddings. `make answer` reuses those indexes and exports `results/financebench/bookrag/gpt-4o-mini/predictions.json`. To force MinerU to a specific backend, pass `MINERU_DEVICE=cpu`, `mps`, or `cuda`.

The standard profile uses `Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-0.6B`. The smoke target uses the smaller cached-compatible `BAAI/bge-reranker-base` while retaining Qwen embeddings. OpenAI is used for graph and answer generation.

### A-RAG

Clone and pin [A-RAG](https://github.com/Ayanami0730/arag), then copy the local harness from [`baselines/arag/`](baselines/arag/README.md) into the clone:

```bash
cd "$ARAG_TOC"
git clone https://github.com/Ayanami0730/arag.git arag
git -C arag checkout a44de6b2216bf6791979c4b6ac4ae106212fa1a6
cp -R baselines/arag/. arag/

cd arag
ln -sf ../.env .env
make setup
```

If `arag/` already exists, verify that it is at commit `a44de6b2216bf6791979c4b6ac4ae106212fa1a6` and copy the harness without cloning it again.

After the BookRAG smoke test, point A-RAG at its MinerU output and run the matching one-case smoke test:

```bash
export FINANCEBENCH_DIR="$ARAG_TOC"
export BOOKRAG_RUNS_DIR="$ARAG_TOC/BookRAG/runs/financebench_qwen_smoke"

make check
make smoke
```

The A-RAG smoke output is:

```text
arag/results/financebench/arag-smoke/gpt-4o-mini/predictions.json
```

For a full run, use the standard BookRAG run directory and keep indexing separate from answering so the same frozen index can be reused across answer models:

```bash
export BOOKRAG_RUNS_DIR="$ARAG_TOC/BookRAG/runs/financebench"

make index DOCS=all
make answer MODEL=gpt-4o-mini
make judge MODEL=gpt-4o-mini ARAG_TOC="$ARAG_TOC"
```

`DEVICE=auto` selects CUDA, then MPS, then CPU. Override it with `DEVICE=cuda`, `cuda:0`, `mps`, or `cpu`. A-RAG writes resumable JSONL during answering and exports the validated artifact to `arag/results/financebench/arag/<model>/predictions.json`.

### Baseline evaluation

Each `make judge` command invokes this repository's `postprocessing.evaluator`, using `gpt-5.4-mini` by default. Override it with `JUDGE_MODEL=<model>` and control concurrency with `PARALLEL=<n>`. The resulting `qa_eval_<timestamp>.json` artifacts use the same schema as the main benchmark and PageIndex results.

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
