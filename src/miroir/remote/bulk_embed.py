# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Embed the stored POP images with A7 and A3 on a Scaleway GPU, unattended (`miroir scw embed`).

The work is cut into 4,096 pieces by the first three hex digits of the file id
(`piece_prefix`). For each piece of its range, the job lists the stored images
(`miroir.jobs.bulk.full_key`), drops those whose vectors July already made (`JULY_LIST_KEY`),
and writes one `.npz` per model to the images bucket (`piece_key`), so a rerun skips the
pieces already written. Each image goes through the July steps: shrunk to 512 px and saved as
JPEG (`miroir.images.fetch.to_jpeg`), opened as RGB, then the models' own encoders
(`miroir.remote.models`). Images are read and prepared on a pool of threads, ahead of the GPU;
A7 runs `A7_STREAMS` batches at once, so the processor's CPU work overlaps the GPU.
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import batched
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from miroir.config import MODAL_WEIGHTS_MOUNT, Settings
from miroir.images.fetch import to_jpeg
from miroir.jobs.bulk import (
    FULL_PREFIX,
    JULY_LIST_KEY,
    JobState,
    JobStatus,
    check_key,
    failures_key,
    full_key,
    piece_key,
    piece_prefix,
    status_key,
)
from miroir.jobs.objects import Bucket, bucket_name, download_tree, listing, s3_client, upload_tree
from miroir.jobs.remote_spec import OPTIONS
from miroir.parallel import bounded_map

logger = logging.getLogger(__name__)

A7 = OPTIONS["A7-embeddinggemma-2-512"]
A3 = OPTIONS["A3-megaloc-322"]
#: Side the July runs embedded (`miroir.images.fetch.LONG_SIDE`).
EMBED_SIDE = 512
READERS = 32
STATUS_EVERY_S = 30.0


def piece_ids(s3, bucket: str, prefix3: str, skip: frozenset[str]) -> list[str]:
    keys = listing(s3, bucket, f"{FULL_PREFIX}/{prefix3[:2]}/{prefix3}")
    return sorted(Path(k.key).stem for k in keys if Path(k.key).stem not in skip)


def prepared(s3, bucket: str, image_id: str) -> PILImage.Image | str:
    """The image as July's models saw it, or why it cannot be read."""
    try:
        data = s3.get_object(Bucket=bucket, Key=full_key(image_id))["Body"].read()
    except Exception as err:
        return f"read: {type(err).__name__}"
    try:
        small = to_jpeg(data, EMBED_SIDE)
        with PILImage.open(io.BytesIO(small)) as opened:
            return opened.convert("RGB")
    except Exception as err:
        return f"unreadable: {type(err).__name__}"


class Models:
    def __init__(self, a7_batch: int, a7_streams: int, a3_batch: int) -> None:
        from miroir.remote.models import MegaLocEncoder, SentenceEncoder  # noqa: PLC0415

        start = time.monotonic()
        # sentence-transformers cuts what it is given into batches of the option's size.
        self.a7 = SentenceEncoder(replace(A7, batch_size=a7_batch))
        self.a3 = MegaLocEncoder(A3)
        self.load_s = time.monotonic() - start
        self.a7_batch = a7_batch
        self.a7_streams = a7_streams
        self.a3_batch = a3_batch
        self._pool = ThreadPoolExecutor(a7_streams)
        self.seconds: dict[str, float] = {"a7": 0.0, "a3": 0.0}

    def run(
        self, images: Sequence[PILImage.Image]
    ) -> tuple[NDArray[np.float16], NDArray[np.float16]]:
        start = time.monotonic()
        a3 = np.concatenate([self.a3.encode(list(b)) for b in batched(images, self.a3_batch)])
        mid = time.monotonic()
        parts = list(
            self._pool.map(lambda b: self.a7.encode(list(b)), batched(images, self.a7_batch))
        )
        a7 = np.concatenate(parts)
        end = time.monotonic()
        self.seconds["a3"] += mid - start
        self.seconds["a7"] += end - mid
        return _unit(a7), _unit(a3)


def _unit(vectors: NDArray[np.float32]) -> NDArray[np.float16]:
    if not np.isfinite(vectors).all():
        msg = "the model returned NaN or infinite values"
        raise FloatingPointError(msg)
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float16)


class Progress:
    """One run's counts, written to its status file (`report`)."""

    def __init__(self, status: JobStatus) -> None:
        self.status = status
        self.started = time.monotonic()
        self.counts: Counter[str] = Counter()
        self.failures: list[str] = []
        self.images = 0
        self.reported = 0.0

    def add(self, ids: Sequence[str], images: Sequence[PILImage.Image | str]) -> list[int]:
        """Count a piece. Returns the rows of the images that could be read."""
        kept: list[int] = []
        for row, (image_id, image) in enumerate(zip(ids, images, strict=True)):
            if isinstance(image, str):
                self.counts[image.split(":")[0]] += 1
                self.failures.append(json.dumps({"image_id": image_id, "reason": image}))
            else:
                kept.append(row)
        self.counts["ok"] += len(kept)
        self.images += len(ids)
        self.status = replace(self.status, done=self.status.done + 1)
        return kept

    def report(self, s3, bucket: str, models: Models, *, force: bool = False) -> None:
        if not force and time.time() - self.reported < STATUS_EVERY_S:
            return
        self.reported = time.time()
        wall = time.monotonic() - self.started
        status = self.status = self.status.now(
            counts=dict(self.counts),
            note=f"{self.images:,} images in {wall:.0f} s ({self.images / max(wall, 1):.1f}/s); "
            f"GPU A7 {models.seconds['a7']:.0f} s, A3 {models.seconds['a3']:.0f} s",
        )
        s3.put_object(Bucket=bucket, Key=status_key(status.job, 0), Body=status.to_json().encode())
        body = "".join(f"{line}\n" for line in self.failures).encode()
        s3.put_object(Bucket=bucket, Key=failures_key(status.job, 0), Body=body)
        logger.info("%s", status.line())


def run_embed(
    settings: Settings,
    *,
    first: int,
    last: int,
    a7_batch: int,
    a7_streams: int,
    a3_batch: int,
    check: int = 0,
    a7_trials: Sequence[tuple[int, int]] = (),
) -> JobStatus:
    """Embed pieces `first` to `last` (inclusive) not yet written."""
    s3 = s3_client(settings)
    bucket = bucket_name(settings, Bucket.IMAGES)
    job = f"embed-{first}-{last}"
    weights = Path(MODAL_WEIGHTS_MOUNT)
    download_tree(s3, bucket_name(settings, Bucket.WEIGHTS), weights)
    july = _july(s3, bucket)
    written = [
        {Path(s.key).stem for s in listing(s3, bucket, f"vectors/{o.id}/")} for o in (A7, A3)
    ]
    done = written[0] & written[1]
    todo = [piece_prefix(p) for p in range(first, last + 1) if piece_prefix(p) not in done]
    note = f"{last - first + 1 - len(todo)} pieces already written"
    progress = Progress(JobStatus(job, 0, 1, total=len(todo), note=note))
    models = Models(a7_batch, a7_streams, a3_batch)
    upload_tree(s3, bucket_name(settings, Bucket.WEIGHTS), weights)
    logger.info("models loaded in %.0f s; %d pieces to do", models.load_s, len(todo))
    if check:
        picked = sorted(july)[:: max(1, len(july) // check)][:check]
        _check(s3, bucket, models, job, picked, trials=a7_trials)

    def load(prefix3: str) -> tuple[str, list[str], list[PILImage.Image | str]]:
        ids = piece_ids(s3, bucket, prefix3, july)
        images = list(bounded_map(lambda i: prepared(s3, bucket, i), ids, workers=READERS))
        return prefix3, ids, images

    progress.started = time.monotonic()
    try:
        # Two pieces are read ahead of the GPU.
        for prefix3, ids, images in bounded_map(load, todo, workers=2, window=2):
            rows = progress.add(ids, images)
            kept = [images[r] for r in rows]
            if kept:
                a7, a3 = models.run([im for im in kept if not isinstance(im, str)])
            else:
                a7 = a3 = np.zeros((0, 1), np.float16)
            kept_ids = np.array([ids[r] for r in rows], dtype=str)
            for option, vectors in ((A7, a7), (A3, a3)):
                buffer = io.BytesIO()
                np.savez(buffer, ids=kept_ids, vectors=vectors)
                key = piece_key(option.id, prefix3)
                s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            progress.report(s3, bucket, models)
    except Exception:
        progress.status = replace(progress.status, state=JobState.FAILED)
        progress.report(s3, bucket, models, force=True)
        raise
    progress.status = replace(progress.status, state=JobState.DONE)
    progress.report(s3, bucket, models, force=True)
    return progress.status


def _july(s3, bucket: str) -> frozenset[str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "july.txt"
        s3.download_file(bucket, JULY_LIST_KEY, str(path))
        return frozenset(path.read_text().split())


def _check(
    s3, bucket: str, models: Models, job: str, ids: list[str], *, trials: Sequence[tuple[int, int]]
) -> None:
    """Embed a few files July embedded, for the Mac to compare with the July vectors, then
    time A7 on them with each (batch, streams) of `trials` and A3 alone."""
    images = list(bounded_map(lambda i: prepared(s3, bucket, i), ids, workers=READERS))
    kept = [(i, im) for i, im in zip(ids, images, strict=True) if not isinstance(im, str)]
    pictures = [im for _, im in kept]
    a7, a3 = models.run(pictures)
    timed: list[tuple[str, int, int, float]] = []
    start = time.monotonic()
    np.concatenate([models.a3.encode(list(b)) for b in batched(pictures, models.a3_batch)])
    timed.append(("A3", models.a3_batch, 1, len(pictures) / (time.monotonic() - start)))
    base = models.a7.option
    for batch, streams in trials:
        models.a7.option = replace(base, batch_size=batch)
        with ThreadPoolExecutor(streams) as pool:
            start = time.monotonic()
            list(pool.map(lambda b: models.a7.encode(list(b)), batched(pictures, batch)))
            timed.append(("A7", batch, streams, len(pictures) / (time.monotonic() - start)))
        logger.info("A7 batch %d, %d stream(s): %.1f images/s", batch, streams, timed[-1][3])
    models.a7.option = base
    buffer = io.BytesIO()
    np.savez(
        buffer,
        ids=np.array([i for i, _ in kept]),
        a7=a7,
        a3=a3,
        timed=np.array([f"{m} {b} {s} {r:.2f}" for m, b, s, r in timed]),
    )
    s3.put_object(Bucket=bucket, Key=check_key(job), Body=buffer.getvalue())
    logger.info("check: %d July files embedded again; %s", len(kept), timed)
