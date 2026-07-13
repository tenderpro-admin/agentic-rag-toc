# A-RAG FinanceBench baseline — gate-ladder convenience targets.
# WCSS-bound steps go through wcss/run_remote.sh (rsync + sbatch --wait + pull).
MODELS ?= gpt-5-mini-2025-08-07 gpt-5-2025-08-07 gpt-4.1-mini-2025-04-14 gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18
WCSS   ?= <your-login>@ui.wcss.pl
REMOTE ?= projects/arag

.PHONY: setup index answer judge status all

setup:            ## one-time: install ST + warm tiktoken on WCSS login node
	bash wcss/run_remote.sh setup

index:            ## build frozen 84-doc Qwen index on WCSS H100
	bash wcss/run_remote.sh index

answer:           ## run the 5-model answer matrix (sequential sbatch --wait)
	@for m in $(MODELS); do echo "== $$m =="; MODEL=$$m bash wcss/run_remote.sh answer; done

judge:            ## judge all models locally in arag-toc -> metrics/arag_<model>.json
	bash wcss/judge_local.sh

status:           ## print the arag-status JSON probe (from WCSS)
	ssh $(WCSS) "cd $(REMOTE) && export WORKDIR=\$$HOME/$(REMOTE) && bash wcss/status.sh"

all: index answer judge   ## full pipeline (index -> answer matrix -> judge)
