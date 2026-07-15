# A-RAG FinanceBench baseline for local macOS and Linux execution.
SHELL := /bin/bash
.DEFAULT_GOAL := help

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_PARENT := $(abspath $(ROOT)/..)
PYTHON := uv run --extra full python

MODEL ?= gpt-4o-mini
MODELS ?= gpt-5-mini-2025-08-07 gpt-5-2025-08-07 gpt-4.1-mini-2025-04-14 gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18
DOCS ?= all
DEVICE ?= auto
BATCH_SIZE ?= 16
EMBED_MODEL ?= Qwen/Qwen3-Embedding-0.6B
MAX_LOOPS ?= 15
LIMIT ?=

BOOKRAG ?= $(REPO_PARENT)/BookRAG
BOOKRAG_RUNS_DIR ?= $(BOOKRAG)/runs/financebench
DEFAULT_FINANCEBENCH_DIR := $(if $(wildcard $(REPO_PARENT)/datasets/finance_bench),$(REPO_PARENT),$(abspath $(ROOT)/../agentic-rag-toc))
FINANCEBENCH_DIR ?= $(DEFAULT_FINANCEBENCH_DIR)
ifeq ($(strip $(FINANCEBENCH_DIR)),)
override FINANCEBENCH_DIR := $(DEFAULT_FINANCEBENCH_DIR)
endif
DATA_DIR ?= $(ROOT)/data/financebench
RESULTS_DIR ?= $(ROOT)/results/financebench/arag
PRED_DIR = $(RESULTS_DIR)/$(MODEL)
PRED_JSONL = $(PRED_DIR)/predictions.jsonl
PRED_JSON = $(PRED_DIR)/predictions.json

ARAG_TOC ?= $(FINANCEBENCH_DIR)
JUDGE_MODEL ?= gpt-5.4-mini
PARALLEL ?= 4

.PHONY: help setup check prepare index answer answer-all export judge smoke all clean

help:
	@echo "A-RAG FinanceBench baseline"
	@echo ""
	@echo "  make setup                    Apply patch and install the local environment"
	@echo "  make check                    Validate local paths, imports, and device"
	@echo "  make index DOCS='all'         Prepare corpora and build frozen indexes"
	@echo "  make answer MODEL='...'       Answer and export predictions"
	@echo "  make answer-all               Run the configured answer-model matrix"
	@echo "  make judge MODEL='...'        Score predictions with the shared evaluator"
	@echo "  make smoke                    Shortest-document index + one answer"
	@echo ""
	@echo "Device selection: DEVICE=auto|cuda|cuda:0|mps|cpu (default: auto)"

setup:
	@cd "$(ROOT)" && test "$$(git rev-parse HEAD)" = "a44de6b2216bf6791979c4b6ac4ae106212fa1a6" || \
		{ echo "ERROR: this harness requires A-RAG commit a44de6b2216bf6791979c4b6ac4ae106212fa1a6" >&2; exit 1; }
	@cd "$(ROOT)" && if git apply --check upstream.patch >/dev/null 2>&1; then \
		git apply upstream.patch; echo "Applied upstream.patch"; \
	elif git apply --reverse --check upstream.patch >/dev/null 2>&1; then \
		echo "upstream.patch already applied"; \
	else echo "ERROR: upstream.patch does not apply to this checkout" >&2; exit 1; fi
	@cd "$(ROOT)" && uv sync --extra full

check:
	@test -f "$(FINANCEBENCH_DIR)/datasets/finance_bench/ground truth/financebench_open_source.jsonl" || \
		{ echo "ERROR: FinanceBench data not found at $(FINANCEBENCH_DIR); set FINANCEBENCH_DIR=/absolute/path/to/agentic-rag-toc" >&2; exit 1; }
	@test -d "$(BOOKRAG_RUNS_DIR)" || \
		{ echo "ERROR: BookRAG runs not found at $(BOOKRAG_RUNS_DIR); set BOOKRAG_RUNS_DIR=/absolute/path/to/BookRAG/runs/<run>" >&2; exit 1; }
	@cd "$(ROOT)" && $(PYTHON) -c 'from financebench.financebench_run import resolve_device; import sentence_transformers; print("device:", resolve_device("$(DEVICE)")); print("sentence-transformers:", sentence_transformers.__version__)'

prepare:
	@cd "$(ROOT)" && $(PYTHON) financebench/financebench_to_arag.py \
		--docs $(DOCS) --runs-dir "$(BOOKRAG_RUNS_DIR)" --out "$(DATA_DIR)" \
		--data-root "$(FINANCEBENCH_DIR)"

index: prepare
	@cd "$(ROOT)" && PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) financebench/financebench_run.py \
		--data-dir "$(DATA_DIR)" --index-only --embed-model "$(EMBED_MODEL)" \
		--device "$(DEVICE)" --batch-size "$(BATCH_SIZE)"

answer:
	@test -f "$(ROOT)/.env" || { echo "ERROR: $(ROOT)/.env missing; copy .env.example and add an API key" >&2; exit 1; }
	@cd "$(ROOT)" && set -a && . ./.env && set +a && \
		export ARAG_API_KEY="$${ARAG_API_KEY:-$${OPENAI_API_KEY:-}}" && \
		if test -z "$$ARAG_API_KEY"; then echo "ERROR: set ARAG_API_KEY or OPENAI_API_KEY in .env" >&2; exit 1; fi; \
		PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) financebench/financebench_run.py \
		--data-dir "$(DATA_DIR)" --out "$(PRED_JSONL)" \
		--embed-model "$(EMBED_MODEL)" --device "$(DEVICE)" \
		--batch-size "$(BATCH_SIZE)" --model "$(MODEL)" --max-loops "$(MAX_LOOPS)" \
		$(if $(LIMIT),--limit "$(LIMIT)",--limit 0)
	@$(MAKE) -C "$(ROOT)" --no-print-directory export MODEL="$(MODEL)" \
		DATA_DIR="$(DATA_DIR)" RESULTS_DIR="$(RESULTS_DIR)" EMBED_MODEL="$(EMBED_MODEL)" LIMIT="$(LIMIT)"

export:
	@cd "$(ROOT)" && $(PYTHON) financebench/export_predictions.py \
		--preds "$(PRED_JSONL)" --model "$(MODEL)" --embed-model "$(EMBED_MODEL)" \
		--out "$(PRED_JSON)" --data-dir "$(DATA_DIR)" \
		$(if $(LIMIT),--limit "$(LIMIT)",--limit 0)

answer-all:
	@for m in $(MODELS); do $(MAKE) -C "$(ROOT)" --no-print-directory answer MODEL="$$m" || exit 1; done

judge:
	@test -f "$(PRED_JSON)" || { echo "ERROR: no predictions at $(PRED_JSON)" >&2; exit 1; }
	@test -f "$(ARAG_TOC)/pyproject.toml" || { echo "ERROR: set ARAG_TOC= to the agentic-rag-toc checkout" >&2; exit 1; }
	@cd "$(ARAG_TOC)" && uv run python -m postprocessing.evaluator "$(PRED_JSON)" \
		--parallel "$(PARALLEL)" --judge-model "$(JUDGE_MODEL)" --results-dir "$(PRED_DIR)"

smoke:
	@$(MAKE) -C "$(ROOT)" --no-print-directory index DOCS=FOOTLOCKER_2022_8K_dated-2022-05-20 \
		DATA_DIR="$(ROOT)/data/financebench-smoke"
	@$(MAKE) -C "$(ROOT)" --no-print-directory answer DOCS=FOOTLOCKER_2022_8K_dated-2022-05-20 LIMIT=1 \
		DATA_DIR="$(ROOT)/data/financebench-smoke" \
		RESULTS_DIR="$(ROOT)/results/financebench/arag-smoke"

all: index answer judge

clean:
	rm -rf "$(DATA_DIR)" "$(RESULTS_DIR)"
