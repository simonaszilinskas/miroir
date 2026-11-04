"""Embedding runs on OpenRouter: probe the cost, embed a gallery or a test set's queries.

The privacy rule (CLAUDE.md rule 8) is enforced here, in `OpenRouterEmbedder`, not by
callers: it loads `publishable` from the record store itself, takes image ids (never image
bytes), and refuses any id whose record is not publishable. Commons photos (T2, T2b, T8c
queries) are openly licensed and pass; they are fetched from Wikimedia by `CommonsImages`.

Money: each request holds its worst case against the budget while in flight and is written to
the ledger as it returns (`miroir.jobs.spend.RunSpend`). Each answer is saved to the run's
checkpoint as it arrives, also the ones still in flight when a run stops, so a stopped run
resumes without paying twice (`miroir.jobs.resumable`).
"""

from __future__ import annotations

import base64
import logging
import random
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import partial
from itertools import batched, chain

import httpx
import numpy as np
from numpy.typing import NDArray

from miroir.commons.images import CommonsImages, split_commons
from miroir.config import SEED, Settings
from miroir.eval.sets import read_gallery, tune_query_images
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
from miroir.jobs.privacy import PublishableImages
from miroir.jobs.resumable import Resumable, kept, unit_float16
from miroir.jobs.spend import (
    PROBE_IMAGES,
    Budget,
    Probe,
    RunSpend,
    SpendKind,
    SpendLedger,
    check_probe,
    check_run,
    load_probe,
    run_limit,
    save_probe,
)
from miroir.parallel import bounded_map
from miroir.records.model import Image

logger = logging.getLogger(__name__)

#: Parallel requests to OpenRouter.
REQUEST_CONCURRENCY = 2
#: A run stops if it costs this many times its projection (prices or billing changed).
OVERRUN_FACTOR = 1.5
#: What one text may cost at most, for holding a text request: a generous ceiling (1,000
#: tokens at $0.10 per million; a search text is a few dozen tokens).
TEXT_USD_EACH = 1e-4


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


@dataclass(frozen=True, slots=True)
class Embedded:
    ids: tuple[str, ...]
    vectors: NDArray[np.floating]
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
        commons: CommonsImages | None = None,
        request_concurrency: int = REQUEST_CONCURRENCY,
        batch_poll_s: float = 30.0,
    ) -> None:
        self.settings = settings
        self.option = option
        self._client = client
        self._fetcher = fetcher
        self._commons = commons
        self._concurrency = request_concurrency
        self._batch_poll_s = batch_poll_s
        self.ledger = SpendLedger(settings.spend_ledger)
        self.budget = Budget.from_settings(settings)
        self._store = PublishableImages.load(settings)
        self._images = self._store.images

    # privacy

    def check_publishable(self, image_ids: Sequence[str]) -> None:
        self._store.check(image_ids, self.option.model)

    def publishable_only(self, image_ids: Sequence[str]) -> list[str]:
        """Drop ids from records that are not publishable (logged), keeping order."""
        return self._store.only_publishable(image_ids)

    # spend

    def spend(
        self, kind: SpendKind, run: str, *, limit_usd: float, per_item_usd: float
    ) -> RunSpend:
        """A run's paid requests, each held against the budget, stopped past `limit_usd`."""
        return RunSpend(
            self.ledger,
            self.budget,
            self.key_usage(),
            limit_usd=limit_usd,
            per_item_usd=per_item_usd,
            kind=kind,
            option=self.option.option,
            run=run,
            model=self.option.model,
        )

    def key_usage(self) -> float | None:
        try:
            return self._client.key_spend().usage_usd
        except (OpenRouterError, httpx.HTTPError):
            logger.warning("could not read the key's spend from OpenRouter; using the ledger")
            return None

    # images

    def embed_images(
        self, image_ids: Sequence[str], spend: RunSpend, job: Resumable | None = None
    ) -> Embedded:
        """Fetch, check and embed images. Stops with BudgetError once `spend` is past its limit.

        With a job, images its checkpoint holds are not sent again, each answer is saved to it
        as it arrives (also those in flight when the run stops), and the result holds the
        checkpoint's vectors as well as the new ones.
        """
        self.check_publishable(image_ids)
        held = set(job.checkpoint.ids) if job else set[str]()
        pop, commons = split_commons(i for i in image_ids if i not in held)
        images = [self._images[i] for i in pop]
        skips: Counter[str] = Counter()
        ids: list[str] = []
        rows: list[NDArray[np.floating]] = []

        def save(answer: tuple[list[Fetched], Embeddings]) -> None:
            chunk_ids = [f.image_id for f in answer[0]]
            if job is not None:
                job.keep(chunk_ids, chunk_ids, answer[1].vectors)
            else:
                rows.append(unit_float16(answer[1].vectors))
            ids.extend(chunk_ids)

        chunks = self._chunks(images, commons, skips)
        embed = partial(self._embed_chunk, spend)
        for answer in bounded_map(embed, chunks, workers=self._concurrency, salvage=save):
            save(answer)
            if len(ids) % 1000 < len(answer[0]):
                todo = len(images) + len(commons)
                logger.info("%s: %d/%d images, $%.4f", spend.run, len(ids), todo, spend.spent_usd)
        if job is not None:
            job.skips.update(skips)
            done, vectors = job.finish(image_ids)
            return Embedded(done, vectors, spend.spent_usd, dict(skips))
        vectors = np.concatenate(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        return Embedded(tuple(ids), vectors, spend.spent_usd, dict(skips))

    def _chunks(
        self, images: Sequence[Image], commons: Sequence[str], skips: Counter[str]
    ) -> Iterator[list[Fetched]]:
        """POP images from their URLs, then Commons photos from Wikimedia, in requests."""
        if commons and self._commons is None:
            msg = f"{len(commons)} Commons photo(s) to embed, but no Commons fetcher was given"
            raise ValueError(msg)
        from_commons = self._commons.fetch_all(commons) if self._commons else iter(())
        both = chain(self._fetcher.fetch_all(images), from_commons)
        fetched = (f for f in both if kept(f, skips))
        for chunk in batched(fetched, self.option.images_per_request):
            yield list(chunk)

    def _embed_chunk(
        self, spend: RunSpend, chunk: list[Fetched]
    ) -> tuple[list[Fetched], Embeddings]:
        inputs = [
            image_input(jpeg_data_url(base64.b64encode(f.jpeg or b"").decode("ascii")))
            for f in chunk
        ]
        with spend.call(len(chunk)) as bill:
            result = self._client.embed(self.option.model, inputs)
            bill.cost_usd = result.cost_usd
        return chunk, result

    # texts

    def embed_texts(self, texts: Mapping[str, str], *, run: str) -> Embedded:
        """Embed texts, through the half-price Batch API when the model has a :batch variant.

        Texts that fail are left out and counted (`batch_failed`)."""
        if not texts:
            return Embedded((), np.zeros((0, 0), dtype=np.float32), 0.0, {})
        batch_model = self._client.batch_model(self.option.model)
        spend = self.spend(
            SpendKind.RUN, run, limit_usd=TEXT_USD_EACH * len(texts), per_item_usd=TEXT_USD_EACH
        )
        ids = list(texts)
        with spend.call(len(ids), model=batch_model) as bill:
            if batch_model is not None:
                inputs: dict[str, str | dict[str, JSON]] = dict(texts)
                vectors, outcome = self._client.embed_batch(
                    batch_model, inputs, poll_s=self._batch_poll_s
                )
                bill.cost_usd = outcome.cost_usd
                failed = {cid: err for cid, err in outcome.errors.items() if cid != "*"}
                if "*" in outcome.errors or failed:
                    logger.warning("%s: %d text(s) failed in batch", run, len(failed) or len(ids))
                ids = [i for i in ids if i in vectors]
                matrix = (
                    np.stack([vectors[i] for i in ids]) if ids else np.zeros((0, 0), np.float32)
                )
            else:
                result = self._client.embed(self.option.model, [texts[i] for i in ids])
                bill.cost_usd = result.cost_usd
                matrix = result.vectors
        skips = {"batch_failed": len(texts) - len(ids)} if len(ids) < len(texts) else {}
        return Embedded(tuple(ids), matrix, spend.spent_usd, skips)


# --- the three entry points --------------------------------------------------------------


def probe(embedder: OpenRouterEmbedder, gallery: str) -> tuple[Probe, int]:
    """Embed 100 images of `gallery`, store the cost per image. Returns it and the gallery size."""
    option = embedder.option
    ids = _gallery(embedder, gallery)
    embedder.check_publishable(ids)
    limit = check_probe(embedder.budget, embedder.ledger, embedder.key_usage())
    sample = random.Random(SEED).sample(ids, min(PROBE_IMAGES, len(ids)))
    run = f"probe-{run_name(option, gallery)}"
    spend = embedder.spend(SpendKind.PROBE, run, limit_usd=limit, per_item_usd=limit / len(sample))
    done = embedder.embed_images(sample, spend)
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
    save_probe(embedder.settings.probe_path(option.option), result)
    logger.info("probe %s: %d images, $%.5f", option.option, result.images, result.cost_usd)
    return result, len(ids)


def embed_gallery(embedder: OpenRouterEmbedder, gallery: str) -> VectorSet:
    """Embed every image of a gallery list and write a vector run (technical spec §7.1)."""
    return _run_images(embedder, _gallery(embedder, gallery), gallery)


def embed_queries(embedder: OpenRouterEmbedder, set_ids: Sequence[str]) -> VectorSet:
    """Embed the query images of test sets' tune halves that may go to an outside service.

    One run holds every set's queries, named after the sets ("T2-T2b-T8c-tune-queries").
    Queries whose image is not publishable are left out; score every option on the same
    remaining queries (technical spec §7). Commons photos are openly licensed and stay.
    """
    ids, label = tune_query_images(embedder.settings, set_ids)
    kept_ids = embedder.publishable_only(ids)
    logger.info("%s: %d of %d query images may be sent", label, len(kept_ids), len(ids))
    return _run_images(embedder, kept_ids, label)


def embed_texts(embedder: OpenRouterEmbedder, name: str, texts: Mapping[str, str]) -> VectorSet:
    """Embed text queries (T6, T7) by id; through the half-price Batch API when it exists.

    Each answer is saved as it arrives. If some texts fail, the run is not written: run it
    again to ask only those.
    """
    run = run_name(embedder.option, name)
    job = Resumable(
        run, embedder.settings.checkpoint_path(run), [embedder.settings.vectors_path(run)]
    )
    start = time.monotonic()
    todo = job.todo(texts)
    done = embedder.embed_texts({i: texts[i] for i in todo}, run=run)
    job.keep(done.ids, done.ids, done.vectors)
    missing = len(job.todo(texts))
    if missing:
        msg = (
            f"{run}: {missing} of {len(texts)} texts failed; the others are saved. "
            "Run it again to ask only those."
        )
        raise OpenRouterError(msg)
    ids, vectors = job.finish(texts)
    paid = embedder.ledger.total(SpendKind.RUN, run=run)
    written = _write(embedder, run, Embedded(ids, vectors, paid, {}), name, start)
    job.checkpoint.remove()
    return written


def _run_images(embedder: OpenRouterEmbedder, ids: list[str], gallery: str) -> VectorSet:
    option = embedder.option
    run = run_name(option, gallery)
    embedder.check_publishable(ids)
    job = Resumable(
        run, embedder.settings.checkpoint_path(run), [embedder.settings.vectors_path(run)]
    )
    todo = job.todo(ids)
    key_usage = embedder.key_usage()
    projected = check_run(
        embedder.budget,
        embedder.ledger,
        key_usage,
        load_probe(embedder.settings.probe_path(option.option)),
        option=option.option,
        model=option.model,
        resolution=LONG_SIDE,
        count=len(todo),
    )
    limit = run_limit(embedder.budget, embedder.ledger, key_usage, projected, factor=OVERRUN_FACTOR)
    logger.info(
        "%s: %d images to send (%d saved before), about $%.2f (stops past $%.2f)",
        run,
        len(todo),
        len(ids) - len(todo),
        projected,
        limit,
    )
    start = time.monotonic()
    spend = embedder.spend(
        SpendKind.RUN, run, limit_usd=limit, per_item_usd=projected / max(len(todo), 1)
    )
    done = embedder.embed_images(ids, spend, job)
    # Cost of the whole run, including the sessions before a resume.
    paid = embedder.ledger.total(SpendKind.RUN, run=run)
    written = _write(
        embedder, run, Embedded(done.ids, done.vectors, paid, dict(job.skips)), gallery, start
    )
    job.checkpoint.remove()
    return written


def _write(
    embedder: OpenRouterEmbedder, run: str, done: Embedded, gallery: str, start: float
) -> VectorSet:
    if done.skips:
        logger.warning("%s: left out %s", run, done.skips)
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
    return write_vectors(embedder.settings.vectors_path(run), done.ids, done.vectors, info)


def _gallery(embedder: OpenRouterEmbedder, name: str) -> list[str]:
    path = embedder.settings.gallery_path(name)
    if not path.exists():
        msg = f"no gallery {name!r} at {path}: run `miroir testsets build` first"
        raise FileNotFoundError(msg)
    return read_gallery(embedder.settings, name)
