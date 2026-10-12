# MIROIR

A visual search engine over French heritage records from POP: photo or words in, matching record out.
Read [docs/functional-spec.md](docs/functional-spec.md), [docs/technical-spec.md](docs/technical-spec.md)
and [docs/experiments.md](docs/experiments.md) before changing behaviour. Code rules: [CODING_STANDARDS.md](CODING_STANDARDS.md).

## Commands

```sh
uv sync                        # base + dev tools (fast; no torch)
uv sync --extra api            # add fastapi, uvicorn
make check                     # ruff format check, ruff lint, pyright strict, pytest. Must pass before commit
make fmt                       # format and auto-fix
uv run miroir --help           # every entry point is a subcommand
uv run miroir records build    # exports -> records.parquet + images.parquet (about 6 s)
uv run miroir testsets build   # records -> test sets T1, T3, T5, T8 + galleries (about 1 s)
uv run miroir testsets commons # Commons photos -> T2, T2b, T8c, screening-v2 (cached; 20 s)
uv run miroir eval run ...      # score a vector run on a test set (technical spec §7.2)
uv run miroir eval stack ...    # fuse models (G1, G2), expand queries (E5), score T8
uv run miroir eval summary     # rebuild the results table in docs/results.md
uv run miroir eval viewer --out page.html   # each model's top 5 as thumbnails, one HTML file
uv run miroir embed modal --option A7-embeddinggemma-2-512 --probe   # open model on a Modal GPU
uv run miroir r0 run           # AmsterTime on Modal; then `miroir eval amstertime score`
uv run miroir vlm ask --model qwen3-vl-30b-a3b --probe   # H1: a vision model describes query photos
uv run miroir vlm score --model qwen3-vl-30b-a3b --image-run <A7 run>   # H1 BM25, H3 mix
```

## Layout

- `src/miroir/config.py`: every setting and path. Ask it for paths; never build one by hand.
- `src/miroir/cli.py`: the `miroir` command. Add subcommands here.
- `src/miroir/records/`: record model (`model.py`), mappers per POP base, Parquet store.
- `src/miroir/remote/`: model code that runs only inside Modal containers. The one place torch may be imported.
- `src/miroir/images/`: image helpers. No local image store: remote jobs read images from their URLs.
- `src/miroir/eval/`: test sets, the vector contract (`vectors.py`), scoring. Results go to `eval/results/`.
- `tests/fixtures/exports/`: real export lines, one per edge case.

## Where data lives

- `MIROIR_EXPORTS_DIR` (default `/Users/simonaszilinskas/Downloads/exports`): POP exports, one folder per base. Read only.
  The Rose Valland folder is spelt `rose-valand`; the collection is `rose-valland`.
- `MIROIR_DATA_DIR` (default `~/miroir-data`): `records.parquet`, `images.parquet`, `vectors/`,
  `testsets/` (tune halves, galleries) and `testsets-sealed/` (sealed halves, kept apart).
- Run models remotely, never on the Mac: OpenRouter if it has the model, else Modal (open models,
  `miroir embed modal`; model code only in `src/miroir/remote/`). Hugging Face Jobs are not approved
  (`OPENROUTER_API_KEY`, `HF_TOKEN` in `.env`). The Mac builds tables and test sets and scores returned vectors.
- Copy `.env.example` to `.env` to change them.
- Load records with `miroir.records.store.read_records` / `read_images`, not by opening Parquet yourself.

## Hard rules

1. Never load or run an ML model on the Mac, not even a smoke test or a 100-image timing: it crashes the 16 GB machine. Never install the `jobs` extra here.
2. Never write into the exports dir.
3. Never commit data, images, Parquet, test sets or model weights. Only small text fixtures.
4. Never read the sealed half of a test set outside round R4. Load sets with
   `miroir.eval.sets.load_queries`; only R4 code may pass it `allow_sealed=True`.
5. Every experiment result goes to `eval/results/<round>/<step>-<option>.json`
   (`config.eval_result_path`), and is committed.
6. No network in unit tests (pytest runs with `--disable-socket`).
7. Paid model runs: probe 100 images first. All OpenRouter spend together stays under $50 (with $5 kept back); `miroir embed openrouter` enforces it. Ask before raising the cap.
8. Only `publishable` records' images may go to outside model services.
9. Start any run longer than a minute fully detached (`nohup uv run miroir … > log 2>&1 &`), then check its log now and then; never wait on it in the foreground. Paid runs save each answer and resume where they stopped.

## Before you commit

`make check` passes. Small commits, one change each. No attribution lines in messages.
