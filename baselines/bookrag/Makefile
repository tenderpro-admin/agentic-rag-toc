# BookRAG FinanceBench baseline for local macOS and Linux execution.
SHELL := /bin/bash
.DEFAULT_GOAL := help

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENV := $(ROOT)/.venv
PYTHON := $(VENV)/bin/python

CONFIG ?= $(ROOT)/config/financebench_gpt4omini.yaml
DATASET_CFG ?= $(ROOT)/Scripts/cfg/financebench.yaml
DATASET_JSON ?= $(ROOT)/datasets/financebench.json
DOCS ?= all
LIMIT ?= 0
MODEL ?= gpt-4o-mini
EMBED_MODEL ?= Qwen/Qwen3-Embedding-0.6B
RUNS_DIR ?= $(ROOT)/runs/financebench
PRED_DIR ?= $(ROOT)/results/financebench/bookrag/$(MODEL)
PRED_OUT ?= $(PRED_DIR)/predictions.json
MINERU_DEVICE ?=
RERANKER_MODEL ?= Qwen/Qwen3-Reranker-0.6B

FINANCEBENCH_DIR ?= $(abspath $(ROOT)/../agentic-rag-toc)
ARAG_TOC ?= $(FINANCEBENCH_DIR)
JUDGE_MODEL ?= gpt-5.4-mini
PARALLEL ?= 4

RUN_ENV = set -a && . "$(ROOT)/.env" && set +a && \
	export PYTHONPATH="$(ROOT)" PYTORCH_ENABLE_MPS_FALLBACK=1 \
	BOOKRAG_LLM_MODEL="$(MODEL)" BOOKRAG_RERANKER_MODEL="$(RERANKER_MODEL)" \
	$(if $(strip $(MINERU_DEVICE)),MINERU_DEVICE_MODE="$(MINERU_DEVICE)",)

.PHONY: help setup check check-env dataset index answer predict judge smoke all clean

help:
	@echo "BookRAG FinanceBench baseline"
	@echo ""
	@echo "  make setup                    Apply patch and create the Python 3.12 environment"
	@echo "  make check                    Validate dependencies, data, and selected device"
	@echo "  make index DOCS='all'         Build the dataset and BookRAG indexes"
	@echo "  make answer MODEL='...'       Answer over existing indexes and export predictions"
	@echo "  make predict                  Dataset, index, and answer"
	@echo "  make judge                    Score predictions with the shared evaluator"
	@echo "  make smoke                    Run the shortest one-case document locally"
	@echo ""
	@echo "Device selection: MINERU_DEVICE=cpu|mps|cuda (unset means auto)"

setup:
	@cd "$(ROOT)" && test "$$(git rev-parse HEAD)" = "113298f919c701d07807ceccb96ed3c18d348117" || \
		{ echo "ERROR: this harness requires BookRAG commit 113298f919c701d07807ceccb96ed3c18d348117" >&2; exit 1; }
	@cd "$(ROOT)" && if git apply --check upstream.patch >/dev/null 2>&1; then \
		git apply upstream.patch; echo "Applied upstream.patch"; \
	elif git apply --reverse --check upstream.patch >/dev/null 2>&1; then \
		echo "upstream.patch already applied"; \
	else echo "ERROR: upstream.patch does not apply to this checkout" >&2; exit 1; fi
	@cd "$(ROOT)" && test -x "$(PYTHON)" || uv venv --python 3.12 "$(VENV)"
	@cd "$(ROOT)" && uv pip install --python "$(PYTHON)" -r requirements-local.txt

check:
	@test -x "$(PYTHON)" || { echo "ERROR: run make setup first" >&2; exit 1; }
	@test -f "$(FINANCEBENCH_DIR)/datasets/finance_bench/ground truth/financebench_open_source.jsonl" || \
		{ echo "ERROR: FinanceBench data not found; set FINANCEBENCH_DIR=" >&2; exit 1; }
	@cd "$(ROOT)" && PYTHONPATH="$(ROOT)" MINERU_DEVICE_REQUESTED="$(MINERU_DEVICE)" "$(PYTHON)" -c 'import os, torch, mineru, transformers, yaml; from Core.configs.system_config import load_system_config; load_system_config("$(CONFIG)"); fail = lambda message: (_ for _ in ()).throw(SystemExit(message)); requested = os.environ["MINERU_DEVICE_REQUESTED"].lower(); requested in ("", "cpu", "mps") or requested.startswith("cuda") or fail("MINERU_DEVICE must be cpu, mps, or cuda"); (not requested.startswith("cuda") or torch.cuda.is_available()) or fail("CUDA requested but unavailable"); (requested != "mps" or hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) or fail("MPS requested but unavailable"); auto = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"; print("device:", requested or auto); print("torch:", torch.__version__)'

check-env:
	@test -f "$(ROOT)/.env" || { echo "ERROR: copy .env.example to .env and add OPENAI_API_KEY" >&2; exit 1; }
	@set -a && . "$(ROOT)/.env" && set +a && test -n "$${OPENAI_API_KEY:-}" || \
		{ echo "ERROR: OPENAI_API_KEY is missing from $(ROOT)/.env" >&2; exit 1; }

dataset:
	@test -x "$(PYTHON)" || { echo "ERROR: run make setup first" >&2; exit 1; }
	@cd "$(ROOT)" && "$(PYTHON)" Scripts/preprocess/financebench_to_bookrag.py \
		--docs $(DOCS) --limit "$(LIMIT)" --out "$(DATASET_JSON)" --data-root "$(FINANCEBENCH_DIR)"

index: dataset check-env
	@cd "$(ROOT)" && $(RUN_ENV) && "$(PYTHON)" main.py -c "$(CONFIG)" -d "$(DATASET_CFG)" \
		--num 1 --nsplit 1 index --stage all

answer: check-env
	@test -f "$(DATASET_JSON)" || { echo "ERROR: dataset missing; run make dataset or make index" >&2; exit 1; }
	@cd "$(ROOT)" && $(RUN_ENV) && "$(PYTHON)" main.py -c "$(CONFIG)" -d "$(DATASET_CFG)" \
		--num 1 --nsplit 1 rag
	@cd "$(ROOT)" && "$(PYTHON)" -m Eval.export_predictions \
		--runs-dir "$(RUNS_DIR)" --dataset "$(DATASET_JSON)" \
		--model "$(MODEL)" --embedding-model "$(EMBED_MODEL)" --out "$(PRED_OUT)"
	@echo "Predictions -> $(PRED_OUT)"

predict: index answer

judge:
	@test -f "$(PRED_OUT)" || { echo "ERROR: no predictions at $(PRED_OUT)" >&2; exit 1; }
	@test -f "$(ARAG_TOC)/pyproject.toml" || { echo "ERROR: set ARAG_TOC= to the agentic-rag-toc checkout" >&2; exit 1; }
	@cd "$(ARAG_TOC)" && uv run python -m postprocessing.evaluator "$(PRED_OUT)" \
		--parallel "$(PARALLEL)" --judge-model "$(JUDGE_MODEL)" --results-dir "$(PRED_DIR)"

smoke:
	@$(MAKE) -C "$(ROOT)" --no-print-directory predict \
		DOCS=FOOTLOCKER_2022_8K_dated-2022-05-20 LIMIT=1 \
		DATASET_CFG="$(ROOT)/Scripts/cfg/financebench_smoke.yaml" \
		DATASET_JSON="$(ROOT)/datasets/financebench_smoke.json" \
		RUNS_DIR="$(ROOT)/runs/financebench_qwen_smoke" \
		RERANKER_MODEL=BAAI/bge-reranker-base \
		PRED_DIR="$(ROOT)/results/financebench/bookrag-smoke/$(MODEL)"

all: predict judge

clean:
	rm -rf "$(RUNS_DIR)" "$(ROOT)/results/financebench/bookrag" \
		"$(ROOT)/results/financebench/bookrag-smoke"
