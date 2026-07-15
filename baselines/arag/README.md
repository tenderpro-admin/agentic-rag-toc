# A-RAG FinanceBench Baseline

Local macOS/Linux reproduction harness for running [A-RAG](https://github.com/Ayanami0730/arag) on FinanceBench. Embedding and indexing support CUDA, Apple MPS, and CPU. Answering uses an OpenAI-compatible endpoint, which may be cloud-hosted or local.

## Automated Setup

Requirements: `git`, `uv`, FinanceBench data, and BookRAG MinerU markdown for the selected documents.

From the `agentic-rag-toc` repository root, run:

```bash
./scripts/setup_arag.sh
```

The script clones and pins A-RAG, copies this overlay, links the repository `.env`, installs dependencies, and checks the root FinanceBench data plus BookRAG's smoke output. It is safe to rerun. Use `./scripts/setup_arag.sh --prepare-only` to skip dependency installation and checks.

## Manual Setup

Run these commands from the `agentic-rag-toc` root:

```bash
git clone https://github.com/Ayanami0730/arag.git arag
git -C arag checkout a44de6b2216bf6791979c4b6ac4ae106212fa1a6
cp -R baselines/arag/. arag/
ln -s ../.env arag/.env
make -C arag setup
make -C arag check \
  FINANCEBENCH_DIR="$(pwd)" \
  BOOKRAG_RUNS_DIR="$(pwd)/BookRAG/runs/financebench_qwen_smoke"
```

Configure `OPENAI_API_KEY`, or `ARAG_API_KEY` and `ARAG_BASE_URL`, in the root `.env` before answering.

## Run

```bash
make -C arag smoke \
  FINANCEBENCH_DIR="$(pwd)" \
  BOOKRAG_RUNS_DIR="$(pwd)/BookRAG/runs/financebench_qwen_smoke"
make -C arag index DOCS=all \
  FINANCEBENCH_DIR="$(pwd)" \
  BOOKRAG_RUNS_DIR="$(pwd)/BookRAG/runs/financebench"
make -C arag answer MODEL=gpt-4o-mini
make -C arag judge MODEL=gpt-4o-mini ARAG_TOC="$(pwd)"
```

Useful alternatives:

```bash
make -C arag answer-all
make -C arag all
```

`DEVICE=auto` chooses CUDA, then MPS, then CPU. Override it with `DEVICE=cuda`, `cuda:0`, `mps`, or `cpu`. `make judge` uses the shared `postprocessing.evaluator` and defaults to `gpt-5.4-mini`.

A-RAG has no PDF parser. `prepare` converts BookRAG's MinerU markdown under `BOOKRAG_RUNS_DIR/<doc_uuid>/auto/<doc_name>.md` into per-document A-RAG corpora, ensuring both baselines use the same parsed text.

## License

Upstream A-RAG ships no license. `upstream.patch` is a minimal research-reproduction change set against pinned commit `a44de6b`. FinanceBench is CC BY-NC 4.0 and is downloaded separately.
