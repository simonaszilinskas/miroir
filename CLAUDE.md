# MIROIR

A visual search engine over French heritage records from POP: photo or words in, matching record out.
Read [docs/functional-spec.md](docs/functional-spec.md), [docs/technical-spec.md](docs/technical-spec.md)
and [docs/experiments.md](docs/experiments.md) before changing behaviour. Code rules: [CODING_STANDARDS.md](CODING_STANDARDS.md).

## Commands

```sh
uv sync                        # base + dev tools (fast; no torch)
uv sync --extra jobs           # torch, transformers: for code that runs inside remote jobs only
uv sync --extra api            # add fastapi, uvicorn
make check                     # ruff format check, ruff lint, pyright strict, pytest. Must pass before commit
make fmt                       # format and auto-fix
uv run miroir --help           # every entry point is a subcommand
uv run miroir records build    # exports -> records.parquet + images.parquet (about 6 s)
uv run miroir testsets build   # records -> test sets T1, T3, T5, T8 + galleries (about 1 s)
```

## Layout

- `src/miroir/config.py`: every setting and path. Ask it for paths; never build one by hand.
- `src/miroir/cli.py`: the `miroir` command. Add subcommands here.
- `src/miroir/records/`: record model (`model.py`), mappers per POP base, Parquet store.
- `src/miroir/images/`: image helpers. No local image store: remote jobs read images from their URLs.
- `src/miroir/eval/`: test sets, scoring. Results go to `eval/results/`.
- `tests/fixtures/exports/`: real export lines, one per edge case.

## Where data lives

- `MIROIR_EXPORTS_DIR` (default `/Users/simonaszilinskas/Downloads/exports`): POP exports, one folder per base. Read only.
  The Rose Valland folder is spelt `rose-valand`; the collection is `rose-valland`.
- `MIROIR_DATA_DIR` (default `~/miroir-data`): `records.parquet`, `images.parquet`, `vectors/`,
  `testsets/` (tune halves, galleries) and `testsets-sealed/` (sealed halves, kept apart).
- Run models remotely, never on the Mac: OpenRouter if it has the model, else Hugging Face Jobs
  (`OPENROUTER_API_KEY`, `HF_TOKEN` in `.env`). The Mac builds tables and test sets and scores returned vectors.
- Copy `.env.example` to `.env` to change them.
- Load records with `miroir.records.store.read_records` / `read_images`, not by opening Parquet yourself.

## Hard rules

1. Never write into the exports dir.
2. Never commit data, images, Parquet, test sets or model weights. Only small text fixtures.
3. Never read the sealed half of a test set outside round R4. Load sets with
   `miroir.eval.sets.load_queries`; only R4 code may pass it `allow_sealed=True`.
4. Every experiment result goes to `eval/results/<round>/<step>-<option>.json`
   (`config.eval_result_path`), and is committed.
5. No network in unit tests (pytest runs with `--disable-socket`).
6. Paid model runs: measure cost on 100 images first; ask before any run over €200.
7. Only `publishable` records' images may go to outside model services.

## Before you commit

`make check` passes. Small commits, one change each. No attribution lines in messages.
