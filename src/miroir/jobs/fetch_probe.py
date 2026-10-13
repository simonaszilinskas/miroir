"""Download speed test of the POP asset bucket on Modal (technical spec §12, question 2).

`miroir images fetch-probe` draws about 1,000 images of the full tables (seeded), then:

1. **Ramp.** One CPU container downloads a fresh batch at 4, 8, 16, 32 and 64 requests at
   once. It stops climbing when more than 2% of requests fail for the server's reasons
   (timeout, connection, 429, 5xx) or the median download takes twice as long as at the
   first step: the last step before that is the safe level.
2. **Spread.** Several containers download at once at the safe level, to see whether the
   bucket holds up when the load comes from many machines.
3. **Copies.** Every image of a few records with several, to find one picture saved twice.

Each download runs once, with no retry, so the failure rate is the raw one. Modal time is
paid by the second and logged to the spend ledger; the run stops before `max_usd`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import modal
import numpy as np

from miroir.config import SEED, Settings
from miroir.images.fetch_probe import TimedBatch, TimedFetch
from miroir.jobs.modal_spec import CPU_CORE_USD_PER_S, MEMORY_GIB_USD_PER_S, Machine
from miroir.jobs.spend import Service, SpendKind, SpendLedger

logger = logging.getLogger(__name__)

APP_NAME = "miroir-fetch-probe"
MACHINE = Machine(gpu=None, cpu=2.0, memory_gib=2)
RAMP = (4, 8, 16, 32, 64)
RAMP_IMAGES = 120
SPREAD_CONTAINERS = 4
SPREAD_IMAGES = 50
#: Records whose every image is fetched, per base, and images kept per record at most.
COPY_RECORDS = {"joconde": 30, "museo": 25}
COPY_IMAGES_PER_RECORD = 4
#: Share of each base in the ramp and spread draws, by the base that first lists the image:
#: enough of each for its own failure rate. Mérimée and Palissy list Mémoire photos, so they
#: come under "memoire"; Palissy's own files are 33 in all.
QUOTA = {
    "memoire": 0.45,
    "joconde": 0.3,
    "rose-valland": 0.1,
    "museo": 0.1,
    "palissy": 0.05,
}
#: Stop climbing past this share of server-side failures, or this slowdown of the median.
MAX_SERVER_FAILURES = 0.02
MAX_SLOWDOWN = 2.0
#: Container start-up and idle time billed around each call, on top of its wall time.
OVERHEAD_S = 30.0
#: The spread step's containers wait this long, so all have started before any downloads.
SPREAD_WAIT_S = 60.0

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "httpx==0.28.1",
        "pillow==12.3.0",
        "numpy==2.5.4",
        "pydantic==2.14.1",
        "pydantic-settings==2.15.0",
    )
    .add_local_python_source("miroir")
)


@app.function(
    image=image,
    cpu=MACHINE.cpu,
    memory=MACHINE.memory_gib * 1024,
    timeout=15 * 60,
    max_containers=SPREAD_CONTAINERS,
    scaledown_window=60,
)
def fetch_batch(
    items: list[tuple[str, str]], concurrency: int, not_before: float = 0.0
) -> TimedBatch:
    from miroir.images.fetch_probe import fetch_timed  # noqa: PLC0415

    return fetch_timed(items, concurrency, not_before)


@dataclass(frozen=True, slots=True)
class Draw:
    ramp: tuple[tuple[tuple[str, str], ...], ...]
    spread: tuple[tuple[tuple[str, str], ...], ...]
    copies: tuple[tuple[str, str], ...]
    base_of: dict[str, str]
    ref_of: dict[str, str]


def draw(settings: Settings, seed: int = SEED) -> Draw:
    """Pick the images: distinct URLs of the full tables, a fixed share from each base."""
    # Imported here: containers load this module too, and their image has no pyarrow.
    from miroir.records.store import read_column  # noqa: PLC0415

    path = settings.full_images_parquet
    ids = np.asarray(read_column(path, "image_id"), dtype=object)
    bases = np.asarray(read_column(path, "collection"), dtype=object)
    refs = np.asarray(read_column(path, "ref"), dtype=object)
    urls = np.asarray(read_column(path, "url"), dtype=object)
    unique = set(settings.full_image_list("unique").read_text(encoding="utf-8").split())
    keep = np.fromiter((i in unique for i in ids), dtype=np.bool_, count=ids.size)
    rng = np.random.default_rng(seed)

    total = len(RAMP) * RAMP_IMAGES + SPREAD_CONTAINERS * SPREAD_IMAGES
    picked: list[int] = []
    for base, share in QUOTA.items():
        rows = np.flatnonzero(keep & (bases == base))
        picked.extend(rng.choice(rows, size=min(rows.size, round(total * share)), replace=False))
    order = rng.permutation(np.asarray(picked))
    pairs = [(str(ids[r]), str(urls[r])) for r in order]
    cut = len(RAMP) * RAMP_IMAGES
    ramp = tuple(tuple(pairs[i : i + RAMP_IMAGES]) for i in range(0, cut, RAMP_IMAGES))
    spread = tuple(
        tuple(pairs[i : i + SPREAD_IMAGES]) for i in range(cut, len(pairs), SPREAD_IMAGES)
    )[:SPREAD_CONTAINERS]

    copies: list[tuple[str, str]] = []
    for base, count in COPY_RECORDS.items():
        rows = np.flatnonzero(keep & (bases == base))
        per_ref: dict[str, list[int]] = {}
        for r in rows:
            per_ref.setdefault(str(refs[r]), []).append(int(r))
        several = sorted(ref for ref, rs in per_ref.items() if len(rs) > 1)
        for ref in rng.choice(np.asarray(several, dtype=object), size=count, replace=False):
            copies.extend(
                (str(ids[r]), str(urls[r])) for r in per_ref[str(ref)][:COPY_IMAGES_PER_RECORD]
            )
    chosen = {int(r) for r in order} | {
        int(r) for r in np.flatnonzero(np.isin(ids, [i for i, _ in copies]))
    }
    return Draw(
        ramp=ramp,
        spread=spread,
        copies=tuple(copies),
        base_of={str(ids[r]): str(bases[r]) for r in chosen},
        ref_of={str(ids[r]): str(refs[r]) for r in chosen},
    )


# --- measures --------------------------------------------------------------------------------


def server_failure(fetch: TimedFetch) -> bool:
    """A failure the server's load could cause: no answer, too many requests, server error."""
    return fetch.status in {0, 429} or fetch.status >= 500


def summarise(batch: TimedBatch) -> dict[str, float | int]:
    """Throughput, failures and sizes of one batch."""
    fetches = batch.fetches
    ok = [f for f in fetches if f.sha1 and not f.error]
    times = np.asarray([f.fetch_s for f in fetches])
    sizes = np.asarray([f.size_bytes for f in ok]) if ok else np.zeros(1)
    return {
        "concurrency": batch.concurrency,
        "images": len(fetches),
        "ok": len(ok),
        "missing": sum(f.status == 404 and f.fallback_status != 200 for f in fetches),
        "rescued_by_iiif": sum(f.status == 404 and f.fallback_status == 200 for f in fetches),
        "server_failures": sum(server_failure(f) for f in fetches),
        "other_failures": sum(
            bool(f.error) and not server_failure(f) and f.status != 404 for f in fetches
        ),
        "wall_s": round(batch.wall_s, 2),
        "images_per_s": round(len(fetches) / batch.wall_s, 2),
        "mb_per_s": round(float(sizes.sum()) / 1e6 / batch.wall_s, 2),
        "median_fetch_s": round(float(np.median(times)), 3),
        "p95_fetch_s": round(float(np.percentile(times, 95)), 3),
        "mean_mb": round(float(sizes.mean()) / 1e6, 3),
        "median_mb": round(float(np.median(sizes)) / 1e6, 3),
        "median_resize_s": round(float(np.median([f.resize_s for f in ok] or [0.0])), 3),
        "mean_resize_s": round(float(np.mean([f.resize_s for f in ok] or [0.0])), 3),
    }


def by_base(fetches: Sequence[TimedFetch], base_of: dict[str, str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for base in QUOTA:
        mine = [f for f in fetches if base_of.get(f.image_id) == base]
        if not mine:
            continue
        ok = [f for f in mine if f.sha1 and not f.error]
        out[base] = {
            "images": len(mine),
            "failed_pct": round(100 * (len(mine) - len(ok)) / len(mine), 1),
            "mean_mb": round(sum(f.size_bytes for f in ok) / max(len(ok), 1) / 1e6, 3),
            "median_megapixels": round(
                float(np.median([f.width * f.height / 1e6 for f in ok] or [0.0])), 2
            ),
            "mean_resize_s": round(float(np.mean([f.resize_s for f in ok] or [0.0])), 3),
        }
    return out


def copy_pairs(fetches: Sequence[TimedFetch], ref_of: dict[str, str]) -> dict[str, int]:
    """Pairs of images of one record that are the same file, or the same picture."""
    per_ref: dict[str, list[TimedFetch]] = {}
    for f in fetches:
        if f.sha1:
            per_ref.setdefault(ref_of[f.image_id], []).append(f)
    pairs = same_file = same_picture = 0
    images_with_twin: set[str] = set()
    for group in per_ref.values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                pairs += 1
                if a.sha1 == b.sha1:
                    same_file += 1
                    images_with_twin.add(b.image_id)
                elif a.phash and b.phash and (a.phash ^ b.phash).bit_count() <= 6:
                    same_picture += 1
                    images_with_twin.add(b.image_id)
    return {
        "records": len(per_ref),
        "images": sum(len(g) for g in per_ref.values()),
        "pairs": pairs,
        "same_file": same_file,
        "same_picture": same_picture,
        "images_that_repeat_another": len(images_with_twin),
    }


# --- the run ---------------------------------------------------------------------------------


@contextmanager
def _open(settings: Settings) -> Generator[None]:
    client = None
    if settings.modal_token_id is not None and settings.modal_token_secret is not None:
        client = modal.Client.from_credentials(
            settings.modal_token_id.get_secret_value(),
            settings.modal_token_secret.get_secret_value(),
        )
    with modal.enable_output(), app.run(client=client):
        yield


class Meter:
    """Modal cost so far: container seconds at the machine's price, plus start-up per call."""

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.usd = 0.0

    def charge(self, wall_s: float, containers: int = 1) -> None:
        self.usd += (wall_s + OVERHEAD_S * containers) * MACHINE.usd_per_s

    def check(self, wall_s_ahead: float, containers: int = 1) -> None:
        ahead = (wall_s_ahead + OVERHEAD_S * containers) * MACHINE.usd_per_s
        if self.usd + ahead > self.max_usd:
            msg = f"next call could pass the ${self.max_usd:.2f} cap (spent ${self.usd:.4f})"
            raise RuntimeError(msg)


def run(settings: Settings, max_usd: float, out_dir: Path) -> dict[str, object]:
    """Run the three steps; write every download and a summary to `out_dir`."""
    if not settings.modal_configured:
        msg = "Modal is not set up: run `uv run modal token new`"
        raise RuntimeError(msg)
    picked = draw(settings)
    logger.info(
        "drew %d ramp, %d spread, %d copy images",
        sum(map(len, picked.ramp)),
        sum(map(len, picked.spread)),
        len(picked.copies),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meter = Meter(max_usd)
    ledger = SpendLedger(settings.spend_ledger)
    every: list[TimedFetch] = []
    summary: dict[str, object] = {"machine": asdict(MACHINE), "seed": SEED}
    start = time.monotonic()
    try:
        with _open(settings):
            ramp, safe = _ramp(picked, meter, every)
            summary["ramp"] = ramp
            summary["safe_concurrency"] = safe
            summary["spread"] = _spread(picked, safe, meter, every)
            meter.check(120.0)
            copies = fetch_batch.remote(list(picked.copies), 16)
            meter.charge(copies.wall_s)
            every.extend(copies.fetches)
            summary["copies"] = copy_pairs(copies.fetches, picked.ref_of)
    finally:
        ledger.record(
            SpendKind.CHECK,
            option="fetch-probe",
            run="full-pop-fetch-probe",
            model="none (CPU downloads)",
            cost_usd=meter.usd,
            items=len(every),
            service=Service.MODAL,
        )
        with (out_dir / "fetches.jsonl").open("w", encoding="utf-8") as out:
            for f in every:
                row = asdict(f) | {"base": picked.base_of.get(f.image_id, "")}
                out.write(json.dumps(row) + "\n")
    summary["by_base"] = by_base(every, picked.base_of)
    summary["cost_usd_estimate"] = round(meter.usd, 4)
    summary["prices"] = {
        "cpu_core_usd_per_s": CPU_CORE_USD_PER_S,
        "gib_usd_per_s": MEMORY_GIB_USD_PER_S,
    }
    summary["elapsed_s"] = round(time.monotonic() - start, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _ramp(
    picked: Draw, meter: Meter, every: list[TimedFetch]
) -> tuple[list[dict[str, float | int]], int]:
    steps: list[dict[str, float | int]] = []
    safe = RAMP[0]
    first_median = 0.0
    for concurrency, batch in zip(RAMP, picked.ramp, strict=False):
        meter.check(RAMP_IMAGES * 2.0 / concurrency + 60)
        result = fetch_batch.remote(list(batch), concurrency)
        meter.charge(result.wall_s)
        every.extend(result.fetches)
        step = summarise(result)
        steps.append(step)
        logger.info("ramp %s", step)
        first_median = first_median or float(step["median_fetch_s"])
        failing = step["server_failures"] / step["images"] > MAX_SERVER_FAILURES
        slower = float(step["median_fetch_s"]) > MAX_SLOWDOWN * first_median
        if failing or slower:
            logger.info("backing off at %d: failures %s, slower %s", concurrency, failing, slower)
            break
        safe = concurrency
    return steps, safe


def _spread(
    picked: Draw, concurrency: int, meter: Meter, every: list[TimedFetch]
) -> dict[str, object]:
    meter.check(SPREAD_IMAGES * 2.0 + SPREAD_WAIT_S, containers=len(picked.spread))
    not_before = time.time() + SPREAD_WAIT_S
    calls = [(list(b), concurrency, not_before) for b in picked.spread]
    batches = list(fetch_batch.starmap(calls))
    for b in batches:
        meter.charge(b.wall_s + SPREAD_WAIT_S)
        every.extend(b.fetches)
    began = min(b.started for b in batches)
    ended = max(b.started + b.wall_s for b in batches)
    fetches = [f for b in batches for f in b.fetches]
    merged = TimedBatch(concurrency, ended - began, began, tuple(fetches))
    return {
        "containers": len(batches),
        "overlap_s": round(
            min(b.started + b.wall_s for b in batches) - max(b.started for b in batches), 1
        ),
        "all": summarise(merged),
        "each": [summarise(b) for b in batches],
    }
