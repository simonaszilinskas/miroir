# Contributing to MIROIR

Working rules for this repository. Read them, with [CODING_STANDARDS.md](CODING_STANDARDS.md),
before changing code. Read [docs/functional-spec.md](docs/functional-spec.md),
[docs/technical-spec.md](docs/technical-spec.md) and [docs/experiments.md](docs/experiments.md)
before changing behaviour. Setup and commands: [docs/developpement.md](docs/developpement.md).

## Layout

- `src/miroir/config.py`: every setting and path. Ask it for paths; never build one by hand.
- `src/miroir/cli.py`: the `miroir` command. Add subcommands here.
- `src/miroir/records/`: record model (`model.py`), mappers per POP base, Parquet store.
- `src/miroir/remote/`: model code that runs only inside Modal containers. The one place torch may
  be imported.
- `src/miroir/images/`: image helpers. No local image store: remote jobs read images from their URLs.
- `src/miroir/jobs/`: paid model work: OpenRouter calls, the Modal app, spend and privacy checks.
- `src/miroir/commons/`, `src/miroir/geo/`, `src/miroir/index/`: Wikimedia Commons photos,
  positions, and the type guess and vector upload.
- `src/miroir/eval/`: test sets, the vector contract (`vectors.py`), scoring. Results go to
  `eval/results/`.
- `src/miroir/web/`: the prototype website (FastAPI on Modal): ranking, card levels, the agentic
  search (`chat.py`, `chat_tools.py`) and the page in `static/`.
- `scripts/pop_full_export.py`: builds the full POP export from public sources.
- `tests/fixtures/exports/`: real export lines, one per edge case.

## Data

- `MIROIR_EXPORTS_DIR`: POP exports, one folder per base. Read only. The Rose Valland folder is
  spelt `rose-valand`; the collection is `rose-valland`.
- `MIROIR_DATA_DIR` (default `~/miroir-data`): `records.parquet`, `images.parquet`, `vectors/`,
  `testsets/` (tune halves, galleries) and `testsets-sealed/` (sealed halves, kept apart).
- Set both in `.env` (copy `.env.example`). API keys (`OPENROUTER_API_KEY`, `HF_TOKEN`) go there
  too and are never committed.
- Load records with `miroir.records.store.read_records` / `read_images`, not by opening Parquet
  yourself.

## Rules

1. Never load or run an ML model on the development machine, not even a smoke test or a short
   timing. Models run on hosted services: OpenRouter when it serves the model, otherwise Modal
   (open models; model code only in `src/miroir/remote/`). The development machine builds tables
   and test sets and scores the vectors that come back. Do not install the `jobs` extra there.
2. Never write into the exports directory.
3. Never commit data, images, Parquet files, test sets or model weights. Only small text fixtures.
4. Never read the sealed half of a test set outside round R4. Load sets with
   `miroir.eval.sets.load_queries`; only R4 code may pass it `allow_sealed=True`.
5. Every experiment result goes to `eval/results/<round>/<step>-<option>.json`
   (`config.eval_result_path`) and is committed.
6. No network in unit tests (pytest runs with `--disable-socket`).
7. Paid model runs start with a probe of 100 images. The CLI enforces a spending cap
   (`MIROIR_OPENROUTER_BUDGET_USD`, see `config.py`); raising it is a project decision.
8. Only images of `publishable` records may go to outside model services
   (`miroir.jobs.privacy`, technical spec §9).
9. Start any run longer than a minute detached (`nohup uv run miroir … > run.log 2>&1 &`) and
   follow its log. Paid runs save each answer and resume where they stopped.

## Commits

`make check` passes before every commit. Small commits, one change each, with a message that says
what changed and why.
