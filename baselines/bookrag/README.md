# BookRAG FinanceBench Baseline

Local macOS/Linux reproduction harness for running [BookRAG](https://github.com/sam234990/BookRAG) on FinanceBench. CUDA, Apple MPS, and CPU are supported; the included profile uses local MinerU, Qwen3-Embedding-0.6B, and Qwen3-Reranker-0.6B plus OpenAI for LLM/VLM generation.

## Automated Setup

Requirements: `git`, `uv`, FinanceBench data, and an `OPENAI_API_KEY`.

From the `agentic-rag-toc` root, the bootstrap script performs the clone, pin, overlay installation, `.env` link, environment setup, and validation:

```bash
./scripts/setup_bookrag.sh
```

It is safe to rerun for a checkout at the expected commit. Use `--prepare-only` to skip dependency installation and checks, or pass a custom destination:

```bash
./scripts/setup_bookrag.sh --prepare-only
./scripts/setup_bookrag.sh /path/to/BookRAG
```

## Manual Setup

Run these commands from the `agentic-rag-toc` root:

```bash
git clone https://github.com/sam234990/BookRAG.git BookRAG
git -C BookRAG checkout 113298f919c701d07807ceccb96ed3c18d348117
cp -R baselines/bookrag/. BookRAG/
ln -s ../.env BookRAG/.env
make -C BookRAG setup
make -C BookRAG check FINANCEBENCH_DIR="$(pwd)"
```

The overlay adds the local harness without replacing upstream's source tree. `FINANCEBENCH_DIR` must contain `datasets/finance_bench/{ground truth,pdfs}`.

## Run

```bash
make -C BookRAG index DOCS=all FINANCEBENCH_DIR="$(pwd)"
make -C BookRAG answer MODEL=gpt-4o-mini
make -C BookRAG judge MODEL=gpt-4o-mini ARAG_TOC="$(pwd)"
```

Useful alternatives:

```bash
make -C BookRAG predict DOCS='BOEING_2022_10K AMD_2022_10K' FINANCEBENCH_DIR="$(pwd)"
make -C BookRAG smoke FINANCEBENCH_DIR="$(pwd)"
make -C BookRAG all FINANCEBENCH_DIR="$(pwd)" ARAG_TOC="$(pwd)"
```

Device selection is automatic. Set `MINERU_DEVICE=cpu`, `mps`, or `cuda` to override MinerU; the Qwen embedder and reranker independently choose CUDA, MPS, then CPU. `make judge` uses the shared `postprocessing.evaluator` from `ARAG_TOC` and defaults to `gpt-5.4-mini`.

The smoke target uses the four-page `FOOTLOCKER_2022_8K_dated-2022-05-20` document and one question. It keeps Qwen embeddings but uses `BAAI/bge-reranker-base`, which follows the standard sequence-classification reranker interface and is more broadly cache-compatible. The full baseline uses `Qwen/Qwen3-Reranker-0.6B`.

Generated datasets, indexes, and results stay in `datasets/`, `runs/`, and `results/` inside the BookRAG clone.

## License

Upstream BookRAG ships no license. `upstream.patch` is a minimal research-reproduction change set against pinned commit `113298f`. FinanceBench is CC BY-NC 4.0 and is downloaded separately.
