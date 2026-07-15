# A-RAG FinanceBench baseline

Local macOS/Linux reproduction harness for running [A-RAG](https://github.com/Ayanami0730/arag) on FinanceBench. Embedding and indexing support CUDA, Apple MPS, and CPU. Answering uses an OpenAI-compatible endpoint, which may be cloud-hosted or local.

## Setup

Requirements: `git`, `uv`, FinanceBench data, and BookRAG MinerU markdown for the selected documents.

1. Clone and pin upstream A-RAG:

   ```bash
   git clone https://github.com/Ayanami0730/arag.git
   cd arag
   git checkout a44de6b2216bf6791979c4b6ac4ae106212fa1a6
   ```

2. Copy the contents of `baselines/arag/` into the clone root. The harness must provide `Makefile`, `upstream.patch`, `financebench/`, and `.env.example`.

3. Install dependencies and configure the answer endpoint:

   ```bash
   make setup
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY, or ARAG_API_KEY and ARAG_BASE_URL.
   export FINANCEBENCH_DIR=/path/to/agentic-rag-toc
   export BOOKRAG_RUNS_DIR=/path/to/BookRAG/runs/financebench
   export ARAG_TOC="$FINANCEBENCH_DIR"
   make check
   ```

## Run

```bash
make index DOCS=all
make answer MODEL=gpt-4o-mini
make judge
```

Useful alternatives:

```bash
make answer-all
make smoke
make all
```

`DEVICE=auto` chooses CUDA, then MPS, then CPU. Override it with `DEVICE=cuda`, `cuda:0`, `mps`, or `cpu`. `make judge` uses the shared `postprocessing.evaluator` and defaults to `gpt-5.4-mini`.

A-RAG has no PDF parser. `prepare` converts BookRAG's MinerU markdown under `BOOKRAG_RUNS_DIR/<doc_uuid>/auto/<doc_name>.md` into per-document A-RAG corpora, ensuring both baselines use the same parsed text.

## License

Upstream A-RAG ships no license. `upstream.patch` is a minimal research-reproduction change set against pinned commit `a44de6b`. FinanceBench is CC BY-NC 4.0 and is downloaded separately.
