.PHONY: check fmt lint types test sync

sync:            ## install base + dev deps
	uv sync

check: lint types test   ## everything CI runs

fmt:             ## rewrite files: format and auto-fix lint
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

types:
	uv run pyright

test:
	uv run pytest
