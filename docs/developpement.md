# MIROIR: développement

How to set up the repository, run the checks and use the `miroir` command. Rules for contributors:
[CLAUDE.md](../CLAUDE.md) (working rules) and [CODING_STANDARDS.md](../CODING_STANDARDS.md)
(code rules).

## Setup

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                  # base, dev tools and the api, modal, openrouter extras (no torch)
cp .env.example .env     # then set the two directories below
make check               # ruff format check, ruff lint, pyright strict, pytest
```

`make check` is what CI runs (`.github/workflows/check.yml`) and must pass before every commit.
`make fmt` formats and applies automatic lint fixes.

## Settings

Every setting is read by `src/miroir/config.py`, from the environment or from `.env`
(see [.env.example](../.env.example)).

| Variable | What it is |
|---|---|
| `MIROIR_EXPORTS_DIR` | POP exports, one folder per base (`joconde`, `memoire`, `rose-valand`). Read only |
| `MIROIR_DATA_DIR` | Where builds, test sets, vectors and indexes go (default `~/miroir-data`). Never inside the repository |
| `OPENROUTER_API_KEY` | Models served by OpenRouter |
| `HF_TOKEN` | Gated model weights, read inside Modal |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal, unless `~/.modal.toml` exists (`modal token new`) |
| `MIROIR_OPENROUTER_BUDGET_USD` | Spending cap for paid model runs (OpenRouter and Modal together) |
| `MIROIR_APP_PASSWORD`, `MIROIR_SESSION_SECRET` | The prototype website's shared password and cookie key (`miroir web secret`) |
| `MIROIR_CHAT_CONVERSATION_USD`, `MIROIR_CHAT_DAY_USD` | Spending caps of the agentic search, per conversation and per day in dollars (defaults 0.10 and 2) |

Data and images are never stored in the repository. Secrets are never committed.

## The `miroir` command

Every entry point is a subcommand: `uv run miroir --help` lists the groups (`records`, `testsets`,
`embed`, `eval`, `web`, …). Models never run on the development machine: embedding, captioning and
re-ranking run on Modal or OpenRouter.

```sh
uv run miroir records build      # exports -> records.parquet + images.parquet (about 6 s)
uv run miroir records build-full # full POP export -> full-pop/ tables, counts, image lists (2 min)
uv run miroir testsets build     # records -> test sets T1, T3, T5, T8 + galleries (about 1 s)
uv run miroir testsets commons   # Commons photos -> T2, T2b, T8c (cached; 20 s)
uv run miroir embed modal --option A7-embeddinggemma-2-512 --probe   # open model on a Modal GPU
uv run miroir eval run ...       # score a vector run on a test set (technical spec §7.2)
uv run miroir eval stack ...     # fuse models (G1, G2), expand queries (E5), score T8
uv run miroir eval summary       # rebuild the results table in docs/results.md
uv run miroir eval viewer --out page.html   # each model's top 5 as thumbnails, one HTML file
uv run miroir web dev            # the website at http://localhost:8000 with fake models
uv run miroir web deploy         # put the website on Modal (steps: prototype.md, "Run and deploy")
```

Paid runs (`embed`, `rerank`, `vlm ask`, `words ask`) start with a probe and stop at a cost
limit (`--probe`, `--max-usd`).
Start any run longer than a minute detached and follow its log:

```sh
nohup uv run miroir records build-full > build-full.log 2>&1 &
```

The full POP export comes from public sources with a standalone script:
`python3 scripts/pop_full_export.py all` (resumable; `--help` for the steps).

## Where to read next

- [Functional specification](functional-spec.md): who it is for, what it does, how we judge it
- [Technical specification](technical-spec.md): data, models, indexes, API, evaluation
- [Experiments](experiments.md): the options tested at each step, and the rounds
- [Results](results.md): every run and what we learned
- [Prototype website](prototype.md): what it does, how it is built and deployed, its scores
