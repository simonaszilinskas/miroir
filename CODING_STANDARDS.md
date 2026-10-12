# Coding standards

Each rule is checkable. Most are enforced by `make check`; the rest by review.

## Structure

1. One package, `src/miroir/`. No scripts outside it; no `if __name__ == "__main__"`.
2. Every entry point is a subcommand of the one `miroir` CLI (`src/miroir/cli.py`, Typer).
   Group by area: `miroir records build`, `miroir jobs embed`, `miroir eval run`.
3. No ML model is ever loaded or run on the Mac, not even for a test: ruff bans importing torch,
   transformers, timm, open_clip, sentence_transformers and mlx. Models run on hosted services.
   The one exception is `src/miroir/remote/`, code that runs only inside Modal containers and is
   imported only inside Modal functions; `tests/test_no_models_on_the_mac.py` checks that.
4. New dependencies: base only if every command needs them. Otherwise an extra
   (`jobs` for remote-job code, `api`) in `pyproject.toml`. Dev tools go in the `dev` group.

## Paths and settings

5. All paths, settings and secrets (`OPENROUTER_API_KEY`, `HF_TOKEN`) come from `miroir.config`.
   No other module reads environment variables, calls `expanduser`/`Path.home`, or hard-codes a
   machine path.
   Ruff bans `os.environ`, `os.getenv` and `os.path`; `tests/test_conventions.py` checks the rest.
6. Need a new path? Add a property or function to `config.py`.

## Types and data

7. Pyright strict passes on `src/`. No `Any` in signatures. `# pyright: ignore[rule]` only for
   untyped third-party calls, with the rule named, on the one line that needs it.
8. Records, images and other values passed between modules are
   `@dataclass(frozen=True, slots=True)`. Tuples, not lists, inside them. Fixed sets of values
   are `StrEnum`.
9. Raw export JSON stays inside `records/mappers.py`. Everything after it uses `Record` / `Image`.
10. Parquet schemas live in `records/store.py`, next to the read and write functions.

## Behaviour

11. Log with `logger = logging.getLogger(__name__)`. No `print` (ruff `T20`). Only the CLI
    configures logging.
12. Randomness takes a seed, defaulting to `config.SEED`: `random.Random(SEED)`,
    `numpy.random.default_rng(SEED)`, `torch.manual_seed(SEED)`. No global unseeded draws.
13. Ids derive from content, never from row order (see `records.images.make_image_id`).
14. Fail loudly: raise on bad input you cannot handle; count and log what you skip, by reason.

## Tests

15. Unit tests need no network (`--disable-socket`), no GPU and no real data dir: use the
    `settings` fixture (fixture exports in, temp dir out).
16. Fixtures are real export lines, copied whole, one per edge case, in `tests/fixtures/`.
17. A bug fix comes with a test that fails before it.

## Evaluation

18. Results are written with `config.eval_result_path(round, step, option)` and committed.
19. Test sets are read only with `miroir.eval.sets.load_queries`. Its sealed half needs
    `allow_sealed=True`, and only R4 code may pass it. `tests/test_conventions.py` fails on any
    other caller; the one exception is `write_set`, which writes the sealed half and never reads it.
