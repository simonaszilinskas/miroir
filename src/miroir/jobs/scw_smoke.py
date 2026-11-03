"""The Scaleway smoke test: embed a few images with A7 and A3, as the gallery runs did, and
check the vectors against the stored ones (cosine above `MIN_COSINE` for every image).

Each option embeds the images twice: the first call fetches them from POP and loads the model,
the second reads them from the machine's disk, which times the GPU alone.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from miroir.config import SEED, Settings
from miroir.eval.vectors import read_vectors
from miroir.jobs.privacy import PublishableImages
from miroir.jobs.remote_spec import OPTIONS
from miroir.jobs.scw_run import ScalewayCalls
from miroir.records.model import Image

logger = logging.getLogger(__name__)

#: Each option and the stored run its vectors must match (made on a Modal L4 in July).
SMOKE_RUNS = {
    "A7-embeddinggemma-2-512": "A7-embeddinggemma-2-512-screening-20k-publishable",
    "A3-megaloc-322": "A3-megaloc-322-screening-20k-publishable",
}
MIN_COSINE = 0.99


@dataclass(frozen=True, slots=True)
class SmokeResult:
    option: str
    asked: int
    embedded: int
    min_cosine: float
    median_cosine: float
    load_s: float
    first_call_s: float
    """Fetch from POP and embed, model already loaded."""
    images_per_s: float
    """Second call: images on the machine's disk, so decode and GPU only."""

    @property
    def ok(self) -> bool:
        return self.embedded == self.asked and self.min_cosine > MIN_COSINE


def smoke_images(settings: Settings, count: int) -> list[Image]:
    """`count` publishable images that every stored smoke run holds, drawn with the seed."""
    held: list[set[str]] = [
        set(read_vectors(settings.vectors_path(r)).ids) for r in SMOKE_RUNS.values()
    ]
    common = sorted(held[0].intersection(*held[1:]))
    ids = random.Random(SEED).sample(common, count)
    store = PublishableImages.load(settings)
    store.check(ids, "Scaleway")
    return [store.images[i] for i in ids]


def run_smoke(calls: ScalewayCalls, settings: Settings, images: list[Image]) -> list[SmokeResult]:
    results: list[SmokeResult] = []
    for option_id, run in SMOKE_RUNS.items():
        opt = OPTIONS[option_id]
        first = calls.embed(opt, images)
        second = calls.embed(opt, images)
        stored = read_vectors(settings.vectors_path(run))
        rows = stored.index()
        mine = first.vectors / np.linalg.norm(first.vectors, axis=1, keepdims=True)
        theirs = np.stack([stored.vectors[rows[i]] for i in first.ids]).astype(np.float32)
        theirs /= np.linalg.norm(theirs, axis=1, keepdims=True)
        cosines = (mine * theirs).sum(axis=1)
        result = SmokeResult(
            option=option_id,
            asked=len(images),
            embedded=len(first.ids),
            min_cosine=float(cosines.min()) if len(cosines) else 0.0,
            median_cosine=float(np.median(cosines)) if len(cosines) else 0.0,
            load_s=first.load_s,
            first_call_s=first.embed_s,
            images_per_s=len(second.ids) / second.embed_s if second.embed_s else 0.0,
        )
        logger.info("%s", result)
        results.append(result)
    return results


def save_smoke(settings: Settings, offer: str, results: list[SmokeResult]) -> Path:
    path = settings.scaleway_dir / f"smoke-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"machine": offer, "results": [{**asdict(r), "ok": r.ok} for r in results]}
    path.write_text(json.dumps(body, indent=2) + "\n")
    return path
