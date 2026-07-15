# BookRAG FinanceBench baseline

Local macOS/Linux reproduction harness for running [BookRAG](https://github.com/sam234990/BookRAG) on FinanceBench. CUDA, Apple MPS, and CPU are supported; the included profile uses local MinerU, Qwen3-Embedding-0.6B, and Qwen3-Reranker-0.6B plus OpenAI for LLM/VLM generation.

## Setup

Requirements: `git`, `uv`, FinanceBench data, and an `OPENAI_API_KEY`.

From the `agentic-rag-toc` root, the bootstrap script performs the clone, pin, overlay installation, `.env` link, environment setup, and validation:

```bash
./scripts/setup_bookrag.sh
```

It is safe to rerun for a checkout at the expected commit. Pass a different destination if needed:

```bash
./scripts/setup_bookrag.sh /path/to/BookRAG
```

Use `--prepare-only` to clone and install the overlay without installing Python dependencies. For manual setup:

1. Clone and pin upstream BookRAG:

   ```bash
   git clone https://github.com/sam234990/BookRAG.git
   cd BookRAG
   git checkout 113298f919c701d07807ceccb96ed3c18d348117
   ```

2. Copy the contents of `baselines/bookrag/` into the clone root. The harness provides `Makefile`, `upstream.patch`, `requirements-local.txt`, `config/`, `Scripts/`, and `Eval/` without replacing upstream's source tree.

3. Create the environment and configure credentials:

   ```bash
   make setup
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY.
   export FINANCEBENCH_DIR=/path/to/agentic-rag-toc
   export ARAG_TOC="$FINANCEBENCH_DIR"
   make check
   ```

`FINANCEBENCH_DIR` must contain `datasets/finance_bench/{ground truth,pdfs}`. It defaults to a sibling `agentic-rag-toc` checkout.

## Run

```bash
make index DOCS=all
make answer MODEL=gpt-4o-mini
make judge
```

Useful alternatives:

```bash
make predict DOCS='BOEING_2022_10K AMD_2022_10K'
make smoke
make all
```

Device selection is automatic. Set `MINERU_DEVICE=cpu`, `mps`, or `cuda` to override MinerU; the Qwen embedder and reranker independently choose CUDA, MPS, then CPU. `make judge` uses the shared `postprocessing.evaluator` from `ARAG_TOC` and defaults to `gpt-5.4-mini`.

The smoke target uses the four-page `FOOTLOCKER_2022_8K_dated-2022-05-20` document and one question. It keeps Qwen embeddings but uses `BAAI/bge-reranker-base`, which follows the standard sequence-classification reranker interface and is more broadly cache-compatible. The full baseline uses `Qwen/Qwen3-Reranker-0.6B`.

Generated datasets, indexes, and results stay in `datasets/`, `runs/`, and `results/` inside the BookRAG clone.

## License

Upstream BookRAG ships no license. `upstream.patch` is a minimal research-reproduction change set against pinned commit `113298f`. FinanceBench is CC BY-NC 4.0 and is downloaded separately.
