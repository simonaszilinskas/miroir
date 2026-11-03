# MIROIR: développement

How to set up the repository, run the checks and use the `miroir` command. Rules for contributors:
[CLAUDE.md](../CLAUDE.md) (working rules) and [CODING_STANDARDS.md](../CODING_STANDARDS.md)
(code rules).

## Setup

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                  # base, dev tools and the api, modal, openrouter, scaleway extras (no torch)
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
| `MISTRAL_API_KEY` | Mistral Large 4 on Mistral's own API, for the agentic search when `OPENROUTER_API_KEY` is not set (OpenRouter wins when both are) |
| `HF_TOKEN` | Gated model weights, read inside Modal or on a Scaleway machine |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal, unless `~/.modal.toml` exists (`modal token new`) |
| `SCW_ACCESS_KEY`, `SCW_SECRET_KEY`, `SCW_DEFAULT_PROJECT_ID`, `SCW_DEFAULT_REGION`, `SCW_DEFAULT_ZONE` | Scaleway: GPU machines and Object Storage (the scw CLI's own names) |
| `MIROIR_SCW_BUDGET_EUR` | Cap on all Scaleway GPU machines together, in euros (default 5); their own ledger, `~/miroir-data/scaleway/spend.jsonl` |
| `MIROIR_SCW_BUCKET_PREFIX` | Start of the bucket names (default `miroir`). Bucket names are shared by every Scaleway customer, so a taken name needs another prefix |
| `MIROIR_OPENROUTER_BUDGET_USD` | Spending cap for paid model runs (OpenRouter and Modal together) |
| `MIROIR_APP_PASSWORD`, `MIROIR_SESSION_SECRET` | The prototype website's shared password and cookie key (`miroir web secret`) |
| `MIROIR_CHAT_CONVERSATION_USD`, `MIROIR_CHAT_DAY_USD` | Spending caps of the agentic search, per conversation and per day in dollars (defaults 0.10 and 2) |

Data and images are never stored in the repository. Secrets are never committed.

## The `miroir` command

Every entry point is a subcommand: `uv run miroir --help` lists the groups (`records`, `testsets`,
`embed`, `eval`, `web`, …). Models never run on the development machine: embedding, captioning and
re-ranking run on Modal, Scaleway or OpenRouter.

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

## Scaleway

The ministry's Scaleway project ("Miroir") now runs the jobs that ran on Modal: GPU machines
and Object Storage. The keys and project id go in `.env` (Settings above). A job, or
the site's search, runs on one GPU machine (an L4 in Paris; in Warsaw, then an L40S, when
Paris has none). Each command starts a machine, builds the job image on it from
`docker/Dockerfile` with this checkout's code (about 4 minutes), runs, and deletes the machine
with its volume and IP: on success, on error and on Ctrl-C. The machine also deletes itself at
`--max-hours`, even if the Mac is gone. Every machine carries the `miroir` tag, so
`miroir scw ps` finds anything left behind and `miroir scw cleanup` deletes it.

Four Object Storage buckets stand in for the Modal volumes (`src/miroir/jobs/objects.py`):

| Bucket | Holds |
|---|---|
| `miroir-weights` | Hugging Face and torch model caches |
| `miroir-benchmarks` | AmsterTime |
| `miroir-images` | POP images at 512 px |
| `miroir-index` | The website's index bundles |

```sh
uv run miroir scw buckets        # create the buckets (once)
uv run miroir scw smoke          # A7 and A3 on 32 images, checked against the stored vectors
uv run miroir embed scw --option A7-embeddinggemma-2-512 --gallery <name> --probe
uv run miroir scw up             # a machine with the worker, until Ctrl-C
uv run miroir scw upload-index   # the site's bundle to the index bucket, then:
uv run miroir web dev --scw      # the local page searching on a Scaleway GPU
uv run miroir scw ps             # machines, loose volumes and IPs left behind; money spent
uv run miroir scw cleanup        # delete them (asks first; --yes does not)
```

`--max-hours` sets a machine's longest life: 1 hour for `scw up`, `scw smoke` and
`web dev --scw`, 2 for `embed scw`.

**Money.** Each machine's life goes to its own ledger, `~/miroir-data/scaleway/spend.jsonl`,
in euros, rounded up to the minute. It is kept apart from the OpenRouter and Modal budget:
another account pays. Before it starts a machine, the command adds the dearest offer's price
for the whole of `--max-hours` to the ledger's total, and refuses when that passes
`MIROIR_SCW_BUDGET_EUR` (default €5). The plan allows €100 in all (search-levels.md); raising
the cap is a project decision.

The prices are typed by hand into `OFFERS` in `src/miroir/jobs/scaleway.py` (euros an hour
without VAT, Scaleway's public prices of November 2026; the API gives none). Check them
against the bill in the Scaleway console, and fix them when they differ.

Smoke test, 2026-11-02, L4-1-24G in fr-par-2: every A7 and A3 vector matched the July Modal
vectors (cosine 1.0000); 6 minutes of machine, €0.08.

All of POP (docs/search-levels.md, "Indexing") runs on machines that need the Mac only to
start and watch them: each writes its progress and failures to the images bucket, and a
rerun skips what is done (`src/miroir/jobs/bulk.py`). Start each one detached.

```sh
uv run miroir scw pop-download --shards 0,1,2,3 --of 8 --upload-list   # images, CPU machine
uv run miroir scw pop-download --shards 4,5,6,7 --of 8                 # a second one
uv run miroir scw pop-embed --first 0 --last 40 --check 64 --upload-list --l40s  # A7 + A3
uv run miroir scw pop-index      # the pop-all bundle, on a machine with 128 GB of memory
uv run miroir scw pop-coverage   # which records have vectors, and why the others do not
```

`pop-download` stores every distinct image file at most 1,024 px on its long side
(`full/<id[:2]>/<id>.jpg`; POP serves no larger file). `pop-embed` embeds the files July did
not, in 4,096 pieces (`--first`, `--last`), with July's preprocessing (512 px JPEG);
`--check` embeds that many July files again to compare. Each machine also deletes itself at
`--max-hours`.

How the first full build ran (2026-11-03), and how to run it again:

```sh
MIROIR_SCW_BUDGET_EUR=62 nohup caffeinate -ims uv run miroir scw pop-embed \
  --first 41 --last 1053 --a7-batch 64 --a7-streams 2 --l40s --max-hours 8 > embed-41-1053.log 2>&1 &
# the same for 1054-2067, 2068-3081 and 3082-4095 (`miroir.jobs.bulk.piece_ranges`)
MIROIR_SCW_BUDGET_EUR=50 nohup caffeinate -ims uv run miroir scw pop-index --max-hours 4 > pop-index.log 2>&1 &
uv run miroir scw pop-coverage   # writes ~/miroir-data/index/coverage-pop-all.json
```

- An L40S embeds 23 images a second with A7 and A3 together; four machines side by side took
  6 h 10 min for 2.0M images, €35 in all. An L4 does about 10 a second for about half the
  price an hour, so it costs more per image.
- `pop-index` runs on a machine with 128 GB of memory (POP2-HM-16C-128G, €1.05 an hour) for
  about 65 minutes: 10 to gather the vectors, 50 to build and score the bundle, 2 to send it.
  The bundle, `pop-all`, lands in the `miroir-index` bucket under `pop-all/` (27.4 GB), and
  the list of files with vectors under `coverage/pop-all.json`.
- Large files go up in parts of at least 8 MB and at most 1,000 parts
  (`miroir.jobs.objects.part_bytes`): Scaleway refuses more, which first stopped the 19 GB
  int8 file.
- Keep the Mac on mains power and awake (`caffeinate -ims`): while it sleeps, finished
  machines wait to be deleted and keep costing.

The full POP export comes from public sources with a standalone script:
`python3 scripts/pop_full_export.py all` (resumable; `--help` for the steps).

## Where to read next

- [Functional specification](functional-spec.md): who it is for, what it does, how we judge it
- [Technical specification](technical-spec.md): data, models, indexes, API, evaluation
- [Experiments](experiments.md): the options tested at each step, and the rounds
- [Results](results.md): every run and what we learned
- [Prototype website](prototype.md): what it does, how it is built and deployed, its scores
