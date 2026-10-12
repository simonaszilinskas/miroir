"""Embedding runs on OpenRouter: probe the cost, embed a gallery or a test set's queries.

The privacy rule (CLAUDE.md hard rule 7) is enforced here, in `OpenRouterEmbedder`, not by
callers: it loads `publishable` from the record store itself, takes image ids (never image
bytes), and refuses any id whose record is not publishable.
"""

from __future__ import annotations

import base64
import logging
import random
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from itertools import batched
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray

from miroir.config import SEED, Settings
from miroir.eval.sets import load_queries, read_gallery
from miroir.eval.vectors import Route, RunInfo, VectorSet, write_vectors
from miroir.images.fetch import LONG_SIDE, Fetched, ImageFetcher
from miroir.jobs.openrouter import (
    JSON,
    Embeddings,
    OpenRouterClient,
    OpenRouterError,
    image_input,
    jpeg_data_url,
)
from miroir.jobs.spend import (
    PROBE_IMAGES,
    Budget,
    BudgetError,
    Probe,
    SpendKind,
    SpendLedger,
    check_probe,
    check_run,
    load_probe,
    save_probe,
)
from miroir.records.model import Image
from miroir.records.store import read_records

logger = logging.getLogger(__name__)

#: Parallel requests to OpenRouter.
REQUEST_CONCURRENCY = 4
#: A run stops if it costs this many times its projection (prices or billing changed).
OVERRUN_FACTOR = 1.5


@dataclass(frozen=True, slots=True)
class EmbedOption:
    option: str
    """Id from experiments.md §2."""
    model: str
    """OpenRouter model id."""
    slug: str
    """Short model name for run folder names."""
    images_per_request: int


OPTIONS: dict[str, EmbedOption] = {
    "A9": EmbedOption("A9", "google/gemini-embedding-2", "gemini-embedding-2", 6),
    "A10a": EmbedOption("A10a", "voyageai/voyage-multimodal-3.5", "voyage-multimodal-3.5", 16),
}


class PrivacyError(PermissionError):
    """Asked to send an image from a record that is not publishable to an outside service."""


@dataclass(frozen=True, slots=True)
class Embedded:
    ids: tuple[str, ...]
    vectors: NDArray[np.float32]
    cost_usd: float
    skips: Mapping[str, int]
    """Inputs left out, by reason."""


def run_name(option: EmbedOption, gallery: str) -> str:
    return f"{option.option}-{option.slug}-{LONG_SIDE}-{gallery}"


class OpenRouterEmbedder:
    """Embeds POP images and texts with one OpenRouter model, and logs every cent it spends."""

    def __init__(
        self,
        settings: Settings,
        option: EmbedOption,
        client: OpenRouterClient,
        fetcher: ImageFetcher,
        *,
        request_concurrency: int = REQUEST_CONCURRENCY,
        batch_poll_s: float = 30.0,
    ) -> None:
        self.settings = settings
        self.option = option
        self._client = client
        self._fetcher = fetcher
        self._concurrency = request_concurrency
        self._batch_poll_s = batch_poll_s
        self.ledger = SpendLedger(settings.spend_ledger)
        self.budget = Budget.from_settings(settings)
        self._images, self._publishable = _load_images(settings)

    # privacy

    def check_publishable(self, image_ids: Sequence[str]) -> None:
        unknown = [i for i in image_ids if i not in self._images]
        if unknown:
            msg = f"{len(unknown)} image id(s) not in images.parquet, e.g. {unknown[0]}"
            raise KeyError(msg)
        private = [i for i in image_ids if i not in self._publishable]
        if private:
            msg = (
                f"{len(private)} image(s) come from records that are not publishable "
                f"(e.g. {private[0]}); they may not go to {self.option.model}"
            )
            raise PrivacyError(msg)

    def publishable_only(self, image_ids: Sequence[str]) -> list[str]:
        """Drop ids from records that are not publishable (logged), keeping order."""
        self.check_publishable([i for i in image_ids if i not in self._images])
        kept = [i for i in image_ids if i in self._publishable]
        if len(kept) < len(image_ids):
            logger.info("left out %d non-publishable image(s)", len(image_ids) - len(kept))
        return kept

    # images

    def embed_images(
        self, image_ids: Sequence[str], *, kind: SpendKind, run: str, limit_usd: float
    ) -> Embedded:
        """Fetch, check and embed images. Stops with BudgetError once `limit_usd` is spent."""
        self.check_publishable(image_ids)
        images = [self._images[i] for i in image_ids]
        skips: Counter[str] = Counter()
        ids: list[str] = []
        rows: list[NDArray[np.float32]] = []
        spent = 0.0
        with ThreadPoolExecutor(self._concurrency) as pool:
            for chunk, result in pool.map(self._embed_chunk, self._chunks(images, skips)):
                spent += result.cost_usd
                self.ledger.record(
                    kind,
                    option=self.option.option,
                    run=run,
                    model=self.option.model,
                    cost_usd=result.cost_usd,
                    items=len(chunk),
                )
                ids.extend(f.image_id for f in chunk)
                rows.append(result.vectors)
                if spent > limit_usd:
                    msg = f"{run}: spent ${spent:.4f}, over its ${limit_usd:.4f} limit; stopped"
                    raise BudgetError(msg)
                if len(ids) % 1000 < len(chunk):
                    logger.info("%s: %d/%d images, $%.4f", run, len(ids), len(images), spent)
        vectors = np.concatenate(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        return Embedded(tuple(ids), vectors, spent, dict(skips))

    def _chunks(self, images: Sequence[Image], skips: Counter[str]) -> Iterator[list[Fetched]]:
        fetched = (f for f in self._fetcher.fetch_all(images) if _keep(f, skips))
        for chunk in batched(fetched, self.option.images_per_request):
            yield list(chunk)

    def _embed_chunk(self, chunk: list[Fetched]) -> tuple[list[Fetched], Embeddings]:
        inputs = [
            image_input(jpeg_data_url(base64.b64encode(f.jpeg or b"").decode("ascii")))
            for f in chunk
        ]
        return chunk, self._client.embed(self.option.model, inputs)

    # texts

    def embed_texts(self, texts: Mapping[str, str], *, run: str) -> Embedded:
        """Embed texts, through the half-price Batch API when the model has a :batch variant."""
        if not texts:
            return Embedded((), np.zeros((0, 0), dtype=np.float32), 0.0, {})
        batch_model = self._client.batch_model(self.option.model)
        ids = list(texts)
        if batch_model is not None:
            inputs: dict[str, str | dict[str, JSON]] = dict(texts)
            vectors, outcome = self._client.embed_batch(
                batch_model, inputs, poll_s=self._batch_poll_s
            )
            cost = outcome.cost_usd
            failed = {cid: err for cid, err in outcome.errors.items() if cid != "*"}
            if "*" in outcome.errors or failed:
                logger.warning("%s: %d text(s) failed in batch", run, len(failed) or len(ids))
            ids = [i for i in ids if i in vectors]
            matrix = np.stack([vectors[i] for i in ids]) if ids else np.zeros((0, 0), np.float32)
            skips = {"batch_failed": len(texts) - len(ids)} if len(ids) < len(texts) else {}
        else:
            result = self._client.embed(self.option.model, [texts[i] for i in ids])
            matrix, cost, skips = result.vectors, result.cost_usd, {}
        self.ledger.record(
            SpendKind.RUN,
            option=self.option.option,
            run=run,
            model=batch_model or self.option.model,
            cost_usd=cost,
            items=len(ids),
        )
        return Embedded(tuple(ids), matrix, cost, skips)

    # spend

    def key_usage(self) -> float | None:
        try:
            return self._client.key_spend().usage_usd
        except (OpenRouterError, httpx.HTTPError):
            logger.warning("could not read the key's spend from OpenRouter; using the ledger")
            return None


# --- the three entry points --------------------------------------------------------------


def probe(embedder: OpenRouterEmbedder, gallery: str) -> tuple[Probe, int]:
    """Embed 100 images of `gallery`, store the cost per image. Returns it and the gallery size."""
    option = embedder.option
    ids = _gallery(embedder, gallery)
    embedder.check_publishable(ids)
    limit = check_probe(embedder.budget, embedder.ledger, embedder.key_usage())
    sample = random.Random(SEED).sample(ids, min(PROBE_IMAGES, len(ids)))
    run = f"probe-{run_name(option, gallery)}"
    done = embedder.embed_images(sample, kind=SpendKind.PROBE, run=run, limit_usd=limit)
    if not done.ids:
        msg = f"probe for {option.option}: no image could be embedded ({done.skips})"
        raise RuntimeError(msg)
    result = Probe(
        option=option.option,
        model=option.model,
        resolution=LONG_SIDE,
        gallery=gallery,
        images=len(done.ids),
        cost_usd=done.cost_usd,
        created=date.today(),  # noqa: DTZ011 (a calendar date for people)
    )
    save_probe(embedder_probe_path(embedder), result)
    logger.info("probe %s: %d images, $%.5f", option.option, result.images, result.cost_usd)
    return result, len(ids)


def embed_gallery(embedder: OpenRouterEmbedder, gallery: str) -> VectorSet:
    """Embed every image of a gallery list and write a vector run (technical spec §7.1)."""
    return _run_images(embedder, _gallery(embedder, gallery), gallery)


def embed_queries(embedder: OpenRouterEmbedder, set_id: str) -> VectorSet:
    """Embed the query images of a test set's tune half that may go to an outside service.

    Queries whose image is not publishable are left out; score every option on the same
    remaining queries (technical spec §7).
    """
    queries = load_queries(embedder.settings, set_id, "tune")
    ids = list(dict.fromkeys(q.query_image_id for q in queries))
    kept = embedder.publishable_only(ids)
    logger.info("%s tune: %d of %d query images are publishable", set_id, len(kept), len(ids))
    return _run_images(embedder, kept, f"{set_id}-tune-queries")


def embed_texts(embedder: OpenRouterEmbedder, name: str, texts: Mapping[str, str]) -> VectorSet:
    """Embed text queries (T6, T7) by id; through the half-price Batch API when it exists."""
    return _run_texts(embedder, texts, name)


def _run_images(embedder: OpenRouterEmbedder, ids: list[str], gallery: str) -> VectorSet:
    option = embedder.option
    run = run_name(option, gallery)
    out = embedder.settings.vectors_path(run)
    _refuse_existing(out)
    embedder.check_publishable(ids)
    projected = check_run(
        embedder.budget,
        embedder.ledger,
        embedder.key_usage(),
        load_probe(embedder_probe_path(embedder)),
        option=option.option,
        model=option.model,
        resolution=LONG_SIDE,
        count=len(ids),
    )
    limit = projected * OVERRUN_FACTOR
    logger.info("%s: %d images, about $%.2f (stops past $%.2f)", run, len(ids), projected, limit)
    start = time.monotonic()
    done = embedder.embed_images(ids, kind=SpendKind.RUN, run=run, limit_usd=limit)
    return _write(embedder, out, done, gallery, start)


def _run_texts(embedder: OpenRouterEmbedder, texts: Mapping[str, str], gallery: str) -> VectorSet:
    run = run_name(embedder.option, gallery)
    out = embedder.settings.vectors_path(run)
    _refuse_existing(out)
    start = time.monotonic()
    done = embedder.embed_texts(texts, run=run)
    return _write(embedder, out, done, gallery, start)


def _write(
    embedder: OpenRouterEmbedder, out: Path, done: Embedded, gallery: str, start: float
) -> VectorSet:
    if done.skips:
        logger.warning("%s: left out %s", out.name, done.skips)
    info = RunInfo(
        option=embedder.option.option,
        model=embedder.option.model,
        resolution=LONG_SIDE,
        pooling="api",
        gallery=gallery,
        created=date.today(),  # noqa: DTZ011
        cost_usd=done.cost_usd,
        wall_time_s=time.monotonic() - start,
        route=Route.OPENROUTER,
        notes=f"skipped: {dict(done.skips)}" if done.skips else "",
    )
    return write_vectors(out, done.ids, done.vectors, info)


def embedder_probe_path(embedder: OpenRouterEmbedder) -> Path:
    return embedder.settings.probe_path(embedder.option.option)


def _refuse_existing(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        msg = f"{out} already exists; a finished run is never overwritten"
        raise FileExistsError(msg)


def _gallery(embedder: OpenRouterEmbedder, name: str) -> list[str]:
    path = embedder.settings.gallery_path(name)
    if not path.exists():
        msg = f"no gallery {name!r} at {path}: run `miroir testsets build` first"
        raise FileNotFoundError(msg)
    return read_gallery(embedder.settings, name)


def _keep(fetched: Fetched, skips: Counter[str]) -> bool:
    if fetched.skip is None:
        return True
    skips[f"fetch_{fetched.skip.value}"] += 1
    return False


def _load_images(settings: Settings) -> tuple[dict[str, Image], frozenset[str]]:
    records = read_records(settings.records_parquet, settings.images_parquet)
    images = {i.image_id: i for r in records for i in r.images}
    publishable = frozenset(i.image_id for r in records if r.publishable for i in r.images)
    return images, publishable
