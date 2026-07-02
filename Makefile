# ---------------------------------------------------------------------------
#  AI Agent — Makefile
#  All commands assume you run from the project root.
#  Ollama is NOT in compose (mac profile) — run it natively on the host.
# ---------------------------------------------------------------------------

.DEFAULT_GOAL := help
SHELL         := /bin/bash
COMPOSE_FILE  := docker/docker-compose.yml
PROFILE       ?= mac        # override: make up PROFILE=gpu-linux
APP_PORT      ?= 8000

# Colours
CYAN  := \033[36m
RESET := \033[0m

.PHONY: help up down logs ps pull-models \
        install lint fmt \
        test test-unit test-int test-cov \
        migrate migration shell-db \
        dev demo eval

## Docker Compose

up: ## Start backing services (PG, Redis, Qdrant) — Ollama runs on host
	docker compose -f $(COMPOSE_FILE) --profile $(PROFILE) up -d --wait
	@echo -e "$(CYAN)✓ Stack up (profile=$(PROFILE)). Ollama must be running natively.$(RESET)"

down: ## Stop and remove containers (keeps volumes)
	docker compose -f $(COMPOSE_FILE) --profile $(PROFILE) down

logs: ## Follow compose logs
	docker compose -f $(COMPOSE_FILE) --profile $(PROFILE) logs -f

ps: ## Show compose service status
	docker compose -f $(COMPOSE_FILE) --profile $(PROFILE) ps

## Models

pull-models: ## Pull required Ollama models (qwen3:14b, bge-m3)
	bash scripts/pull_models.sh

## Python / uv

install: ## Install all dependencies (including dev) via uv
	uv sync --all-extras

## Lint & Format

lint: ## Run ruff check + mypy strict
	uv run ruff check app tests
	uv run mypy app

fmt: ## Auto-format and fix lint issues
	uv run ruff format app tests
	uv run ruff check --fix app tests

## Tests

test: ## Run all tests
	uv run pytest

test-unit: ## Run unit tests only (no I/O)
	uv run pytest -m unit

test-int: ## Run integration tests (require running services)
	uv run pytest -m integration

test-cov: ## Run tests with coverage report
	uv run pytest --cov=app --cov-report=term-missing --cov-report=html

## Migrations

migrate: ## Apply all pending Alembic migrations
	uv run alembic upgrade head

migration: ## Generate a new migration (MSG="describe change")
	uv run alembic revision --autogenerate -m "$(MSG)"

shell-db: ## Open psql in the running Postgres container
	docker compose -f $(COMPOSE_FILE) exec postgres psql -U agent -d agent

## Dev server

dev: ## Start FastAPI in reload mode
	uv run uvicorn app.main:app --reload --port $(APP_PORT) --log-level info

## Demo & Eval

demo: ## Run scripted end-to-end demo
	uv run python scripts/demo.py

eval: ## Run full evaluation suite (RAG + agents + routing), incl. local Ollama
	uv run python scripts/eval_run.py

## Help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2}'
