# BookRAG x FinanceBench pipeline (mirrors arag-toc's PageIndex Makefile/DVC).
#
# Flow:  dataset -> index (frozen BookIndex) -> answer (rag + export predictions)
#        -> judge (the SAME gpt-5-mini judge that grades PageIndex, in arag-toc).
#
# Secrets: OPENAI_API_KEY is loaded ONLY from .env (gitignored). The judge runs
# in the arag-toc repo so both systems are scored by the identical judge.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# --- knobs (override on the CLI, e.g. `make predict DOCS=AMD_2022_10K`) ---
CONFIG       ?= config/financebench_gpt4omini.yaml
DATASET_CFG  ?= Scripts/cfg/financebench_boeing.yaml
DATASET_JSON ?= datasets/financebench_boeing_smoke.json
DOCS         ?= BOEING_2022_10K
MODEL        ?= gpt-4o-mini
RUNS_DIR     ?= runs/financebench
PRED_DIR     ?= results/financebench/bookrag/$(MODEL)
PRED_OUT     ?= $(PRED_DIR)/predictions.json
MINERU_DEVICE?= mps
# FinanceBench data root (holds datasets/finance_bench/...).
FINANCEBENCH_DIR ?= $(HOME)/finance_bench_root
# arag-toc checkout that owns the shared judge (run judging there).
ARAG_TOC     ?= ../arag-toc
JUDGE_MODEL  ?= gpt-5-mini-2025-08-07
PARALLEL     ?= 4

LOAD_ENV = set -a && . ./.env && set +a
PYRUN    = source .venv/bin/activate && $(LOAD_ENV) && export PYTHONPATH=. MINERU_DEVICE_MODE=$(MINERU_DEVICE) PYTORCH_ENABLE_MPS_FALLBACK=1

# --- WCSS (Lem GPU / H100) knobs ---
WCSS_HOST   ?= <your-login>@ui.wcss.pl
WCSS_REMOTE ?= projects/BookRAG

.PHONY: help check-env dataset index answer predict judge smoke clean-runs \
        wcss-setup wcss-index wcss-answer wcss-judge wcss-smoke wcss-status

help:
	@echo "BookRAG x FinanceBench (gpt-4o-mini). Secrets from .env only."
	@echo ""
	@echo "  make dataset DOCS='A B C'   Build BookRAG dataset JSON from FinanceBench docs"
	@echo "  make index                  Build the frozen BookIndex (tree+graph+vdb) per doc"
	@echo "  make answer                 Run RAG + export predictions.json (shared schema)"
	@echo "  make predict                dataset -> index -> answer (end to end)"
	@echo "  make judge                  Score predictions with the arag-toc gpt-5-mini judge"
	@echo "  make smoke                  BOEING_2022_10K end-to-end (predict + judge)"
	@echo ""
	@echo "Knobs: DOCS=, MODEL=, MINERU_DEVICE=mps|cpu|cuda, FINANCEBENCH_DIR=, ARAG_TOC="

check-env:
	@test -f .env || { echo "ERROR: .env not found. Add OPENAI_API_KEY=... to ./.env"; exit 1; }

dataset:
	@source .venv/bin/activate && FINANCEBENCH_DIR=$(FINANCEBENCH_DIR) \
	  python Scripts/preprocess/financebench_to_bookrag.py --docs $(DOCS) --out $(DATASET_JSON)

# Frozen index: built once per doc set. Re-run only when docs/config change.
index: check-env
	@$(PYRUN) && python main.py -c $(CONFIG) -d $(DATASET_CFG) \
	  --num 1 --nsplit 1 index --stage all

answer: check-env
	@$(PYRUN) && python main.py -c $(CONFIG) -d $(DATASET_CFG) \
	  --num 1 --nsplit 1 rag
	@$(PYRUN) && python -m Eval.export_predictions \
	  --runs-dir $(RUNS_DIR) --model $(MODEL) --out $(PRED_OUT)
	@echo "Predictions -> $(PRED_OUT)"

predict: dataset index answer

# Score with the SAME judge PageIndex uses (lives in the arag-toc repo).
# Copies predictions into arag-toc's results tree, then runs its judge there.
judge: check-env
	@test -d "$(ARAG_TOC)" || { echo "ERROR: arag-toc not found at $(ARAG_TOC) (set ARAG_TOC=)"; exit 1; }
	@mkdir -p "$(ARAG_TOC)/results/financebench/bookrag/$(MODEL)"
	@cp "$(PRED_OUT)" "$(ARAG_TOC)/results/financebench/bookrag/$(MODEL)/predictions.json"
	@cd "$(ARAG_TOC)" && set -a && . ./.env && set +a && \
	  python -m eval.judge_predictions \
	    results/financebench/bookrag/$(MODEL)/predictions.json \
	    --judge-model $(JUDGE_MODEL) --parallel $(PARALLEL) \
	    --output results/financebench/bookrag/$(MODEL)/judged.json \
	    --metrics-out metrics/bookrag_$(MODEL).json
	@echo "Judged -> $(ARAG_TOC)/results/financebench/bookrag/$(MODEL)/judged.json"

smoke: predict judge

clean-runs:
	rm -rf $(RUNS_DIR)/* $(PRED_DIR)

# ----------------------------------------------------------------------------
# WCSS (Lem GPU / H100) targets — the real indexing path. index + answer run on
# the cluster via wcss/run_remote.sh (push -> sbatch --wait -> pull); judge runs
# locally with the shared arag-toc gpt-5-mini judge. Driven by params.yaml too:
#   uv run dvc repro            # index -> answer -> judge on WCSS
# ----------------------------------------------------------------------------

# One-time: build the conda env + pre-download all models on the WCSS login node.
wcss-setup:
	rsync -az --exclude='.git/' --exclude='.venv/' --exclude='runs/' \
	  --exclude='.cache/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pdf' \
	  ./ $(WCSS_HOST):~/$(WCSS_REMOTE)/
	ssh $(WCSS_HOST) 'cd ~/$(WCSS_REMOTE) && bash wcss/setup_env.sh'

# Build the frozen BookIndex on the H100 (dataset -> MinerU GPU -> tree/graph/vdb).
wcss-index:
	DOCS='$(DOCS)' wcss/run_remote.sh index

# Answer over the frozen index + export predictions (H100 reranker/embeddings).
wcss-answer:
	ANSWER_MODEL='$(MODEL)' DOCS='$(DOCS)' wcss/run_remote.sh rag

# Score with the SAME judge PageIndex uses (arag-toc gpt-5-mini).
wcss-judge:
	ANSWER_MODEL='$(MODEL)' JUDGE_MODEL='$(JUDGE_MODEL)' PARALLEL='$(PARALLEL)' \
	  ARAG_TOC='$(ARAG_TOC)' wcss/judge_local.sh

# BOEING single-doc smoke, end to end on WCSS.
wcss-smoke:
	DOCS=BOEING_2022_10K SMOKE=1 wcss/run_remote.sh index
	ANSWER_MODEL='$(MODEL)' DOCS=BOEING_2022_10K SMOKE=1 wcss/run_remote.sh rag
	$(MAKE) wcss-judge MODEL='$(MODEL)'

wcss-status:
	ssh $(WCSS_HOST) 'squeue -u $$USER'
