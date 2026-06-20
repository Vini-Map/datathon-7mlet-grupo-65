# Makefile — single source of truth for project commands.
# On Windows without `make`, use the equivalent PowerShell wrapper: ./make.ps1 <target>
# (e.g. `./make.ps1 setup`). Each target maps 1:1 to a wrapper command.

UV ?= uv

.DEFAULT_GOAL := help
.PHONY: help setup setup-all data train eval serve test lint format demo clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install core + dev dependencies
	$(UV) sync

setup-all:  ## Install ALL dependency groups (eda, ml, service, rag, llm)
	$(UV) sync --all-groups

data:  ## (Stage 1-2) Download/process Kaggle base and generate synthetic enrichment
	$(UV) run aep data --help

train:  ## (Stage 3) Train/simulate bandit policies and log to MLflow
	$(UV) run aep train --help

eval:  ## (Stage 4) Run reproducible offline evaluation against the golden set
	$(UV) run aep eval --help

serve:  ## (Stage 5) Start the FastAPI decision service
	$(UV) run aep serve --help

demo:  ## (Stage 5) Run the end-to-end pipeline locally (data -> train -> eval -> decision)
	$(UV) run aep demo --help

test:  ## Run the test suite
	$(UV) run pytest

lint:  ## Lint with ruff
	$(UV) run ruff check src tests

format:  ## Format with black + ruff import sort
	$(UV) run black src tests
	$(UV) run ruff check --fix src tests

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info htmlcov
