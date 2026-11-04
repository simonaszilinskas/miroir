.PHONY: check fmt lint types test sync ui

sync:            ## install base + dev deps
	uv sync

check: lint types test   ## everything CI runs

fmt:             ## rewrite files: format and auto-fix lint
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

lint:
	uv run ruff format --check src tests scripts
	uv run ruff check src tests scripts

types:
	uv run pyright

test:
	uv run pytest

ui:              ## build the site's page (web-ui/, Node) into src/miroir/web/ui/
	npm --prefix web-ui ci
	npm --prefix web-ui run build
