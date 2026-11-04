"""Embedding runs on the GPU machine, driven from the Mac: probe, gallery run, query run.

Every call goes to one Scaleway GPU machine (`miroir.jobs.scw_run.open_scaleway`), which
yields the calls (`RemoteCalls`) and the machine's price (`RemoteCalls.machine`).

Money (experiments.md §1 rule 5). Each remote call is timed on the Mac, from the call to its
answer, and costs that many seconds at the machine's price: a little more than the machine's
share of the bill, since the clock also runs while the call travels. The cost goes into the one
project ledger, tagged with the service (`scaleway`), even when the call fails. While a call
runs, its worst case (its timeout at full price) is held against the usable budget
(`SpendLedger.hold`). A gallery run needs a 100-image probe of the same model and size on file,
and stops once it costs `OVERRUN_FACTOR` times its projection (`run_limit`). Each answer is
checked against the ids sent and saved to a checkpoint as it arrives, so a stopped run resumes
without paying twice (`miroir.jobs.resumable`).

Privacy (technical spec §9): only images of `publishable` records go to the GPU machine. The
calls check every image themselves (`RemoteCalls.gate`, an `ImageGate`), and runs check whole
galleries before paying. Commons query photos are openly licensed: the Mac fetches them from
Wikimedia (`CommonsImages`) and sends their bytes. The full-sample index is an exception
(`all_images` on the gate): open models on our own machines may see every image, publishable or
not (decided 2026-10-12 for the prototype). It never applies to OpenRouter.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from itertools import batched
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter, ValidationError

from miroir.commons.images import CommonsImages, split_commons
from miroir.config import SEED, Settings
from miroir.eval.sets import read_gallery, tune_query_images
from miroir.eval.vectors import Route, RunInfo, VectorSet, read_vectors, write_vectors
from miroir.images.fetch import Fetched
from miroir.jobs.privacy import ImageGate
from miroir.jobs.remote_spec import (
    CHUNK_IMAGES,
    CHUNK_TEXTS,
    CHUNK_TIMEOUT_S,
    FILL_TIMEOUT_S,
    OPTIONS,
    TEXT_PROMPT,
    ChunkResult,
    Family,
    Machine,
    ModelOption,
    Pooling,
    SuppliedImage,
    option,
)
from miroir.jobs.resumable import Resumable, check_answer, kept, refuse_existing
from miroir.jobs.spend import (
    PROBE_IMAGES,
    Bill,
    Budget,
    BudgetError,
    RunSpend,
    Service,
    SpendKind,
    SpendLedger,
    check_probe,
    check_run,
    run_limit,
)
from miroir.records.model import Image

logger = logging.getLogger(__name__)

#: A run stops once it costs this many times its projection, plus the margin below.
OVERRUN_FACTOR = 1.5
#: Start-up varies more than per-image time; small runs get this much slack on top.
OVERRUN_MARGIN_USD = 0.25


class SuppliesImages(Protocol):
    """Gives image bytes by id, in order: Commons photos, extra records (`embed_supplied`)."""

    def fetch_all(self, image_ids: Iterable[str]) -> Iterator[Fetched]: ...


class RemoteCalls(Protocol):
    """The calls a run makes. `miroir.jobs.scw_run.ScalewayCalls` sends them to the machine."""

    @property
    def machine(self) -> Machine:
        """The machine every call runs on, at its price."""
        ...

    @property
    def gate(self) -> ImageGate:
        """The privacy check each call that sends an image runs before sending."""
        ...

    def embed(
        self,
        option: ModelOption,
        images: Sequence[Image],
        supplied: Sequence[SuppliedImage] = (),
    ) -> ChunkResult: ...
    def embed_texts(
        self, option: ModelOption, ids: Sequence[str], texts: Sequence[str]
    ) -> ChunkResult: ...
    def fill_images(self, chunks: Sequence[Sequence[Image]]) -> dict[str, int]: ...


@dataclass(frozen=True, slots=True)
class GpuProbe:
    """Time and cost of embedding about 100 gallery images with one option on a GPU."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    option: str
    model: str
    resolution: int
    gallery: str
    images: int
    fixed_s: float
    """Seconds of the call not spent embedding: start-up, model download and load."""
    per_image_s: float
    usd_per_s: float
    cost_usd: float
    created: date

    def projected_usd(self, count: int) -> float:
        return (self.fixed_s + self.per_image_s * count) * self.usd_per_s


_PROBE = TypeAdapter(GpuProbe)


def save_gpu_probe(path: Path, probe: GpuProbe) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PROBE.dump_json(probe, indent=2) + b"\n")


def load_gpu_probe(path: Path) -> GpuProbe | None:
    if not path.exists():
        return None
    try:
        return _PROBE.validate_json(path.read_bytes())
    except (ValidationError, json.JSONDecodeError) as err:
        msg = f"{path} is not a GPU probe: {err}"
        raise BudgetError(msg) from err


def run_name(opt: ModelOption, gallery: str) -> str:
    return f"{opt.id}-{gallery}"


def output_runs(opt: ModelOption, gallery: str) -> dict[str, Pooling]:
    """The vector runs a gallery run writes, with their pooling. CLS+GeM writes two."""
    if opt.pooling is Pooling.CLS_GEM:
        return {f"{opt.id}-{p.value}-{gallery}": p for p in (Pooling.CLS, Pooling.GEM)}
    return {run_name(opt, gallery): opt.pooling}


@dataclass(frozen=True, slots=True)
class Paid[T]:
    result: T
    cost_usd: float
    wall_s: float


@dataclass(frozen=True, slots=True)
class PaidCall:
    """What a paid call on the GPU machine needs to check the budget and log its cost."""

    ledger: SpendLedger
    budget: Budget
    key_usage_usd: float | None = None
    clock: Callable[[], float] = time.monotonic
    service: Service = Service.SCALEWAY


def paid_call[T](
    paid: PaidCall,
    call: Callable[[], T],
    *,
    machine: Machine,
    timeout_s: int,
    kind: SpendKind,
    option: str,
    run: str,
    model: str,
    items: int,
) -> Paid[T]:
    """Make one remote call with its worst case held against the budget; log its cost, even
    when it fails."""
    worst = machine.usd_per_s * timeout_s
    hold = paid.ledger.hold(
        paid.budget,
        paid.key_usage_usd,
        worst_usd=worst,
        expected_usd=worst,
        kind=kind,
        option=option,
        run=run,
        model=model,
        items=items,
        service=paid.service,
    )
    return timed(hold, call, machine, paid.clock)


def timed[T](
    hold: AbstractContextManager[Bill],
    call: Callable[[], T],
    machine: Machine,
    clock: Callable[[], float] = time.monotonic,
) -> Paid[T]:
    """Run `call` inside a hold. It costs its wall time at the machine's price."""
    with hold as bill:
        start = clock()
        try:
            result = call()
        finally:
            wall = clock() - start
            bill.cost_usd = wall * machine.usd_per_s
    return Paid(result, wall * machine.usd_per_s, wall)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One call's images: POP images the machine reads itself, and images sent as bytes."""

    images: Sequence[Image] = ()
    supplied: Sequence[SuppliedImage] = ()

    @property
    def ids(self) -> list[str]:
        return [*(i.image_id for i in self.images), *(s.image_id for s in self.supplied)]


#: Turns the ids still to embed into the calls' chunks, counting images it could not fetch.
type Chunker = Callable[[Sequence[str], Counter[str]], Iterator[Chunk]]


class GpuRunner:
    """Runs embedding jobs on the GPU machine and logs what each call costs."""

    def __init__(
        self,
        settings: Settings,
        calls: RemoteCalls,
        *,
        key_usage_usd: float | None = None,
        commons: CommonsImages | None = None,
        clock: Callable[[], float] = time.monotonic,
        service: Service = Service.SCALEWAY,
    ) -> None:
        self.settings = settings
        self._service = service
        self._calls = calls
        self._commons = commons
        """Fetches Commons query photos on the Mac; needed only to embed them."""
        self._key_usage = key_usage_usd
        """What OpenRouter says its key has spent, read once by the caller (or None)."""
        self._clock = clock
        self.ledger = SpendLedger(settings.spend_ledger)
        self.budget = Budget.from_settings(settings)

    # --- the entry points -------------------------------------------------------------------

    def probe(self, option_id: str, gallery: str) -> tuple[GpuProbe, int]:
        """Embed 100 images of `gallery`, store time and cost. Returns it and the gallery size."""
        opt = option(option_id)
        ids = self._gallery(gallery)
        left = check_probe(self.budget, self.ledger, self._key_usage)
        worst = self._calls.machine.usd_per_s * CHUNK_TIMEOUT_S
        if worst > left:
            msg = f"a probe may cost up to ${worst:.2f}; probes have ${left:.2f} left"
            raise BudgetError(msg)
        sample = random.Random(SEED).sample(ids, min(PROBE_IMAGES, len(ids)))
        run = f"probe-{run_name(opt, gallery)}"
        spend = self._spend(SpendKind.PROBE, opt.id, run, opt.model, limit_usd=left)
        paid = self._timed(spend, lambda: self._calls.embed(opt, self._images(sample)), sample)
        done = paid.result
        if not done.ids:
            msg = f"probe for {opt.id}: no image could be embedded ({done.skips})"
            raise RuntimeError(msg)
        check_answer(run, sample, done.ids, done.vectors)
        result = GpuProbe(
            option=opt.id,
            model=opt.model,
            resolution=opt.resolution,
            gallery=gallery,
            images=len(done.ids),
            fixed_s=max(paid.wall_s - done.embed_s, 0.0),
            per_image_s=done.embed_s / len(done.ids),
            usd_per_s=self._calls.machine.usd_per_s,
            cost_usd=paid.cost_usd,
            created=_today(),
        )
        save_gpu_probe(self.settings.probe_path(opt.id), result)
        logger.info(
            "probe %s: %d images, $%.4f (load %.0f s, %.3f s per image), skipped %s",
            opt.id,
            result.images,
            result.cost_usd,
            done.load_s,
            result.per_image_s,
            done.skips,
        )
        return result, len(ids)

    def embed_gallery(
        self,
        option_id: str,
        gallery: str,
        *,
        max_usd: float | None = None,
        reuse: Sequence[str] = (),
    ) -> tuple[VectorSet, ...]:
        """Embed every image of a gallery and write its vector runs (technical spec §7.1).

        One run, or two for CLS+GeM (`output_runs`). `max_usd` refuses the run when the probe
        projects a higher cost. `reuse` names vector runs of the same model and settings whose
        vectors are taken as they are instead of being paid for again.
        """
        opt = option(option_id)
        ids = self._gallery(gallery)

        def chunks(todo: Sequence[str], _skips: Counter[str]) -> Iterator[Chunk]:
            for chunk in _chunks(todo):
                yield Chunk(images=self._images(chunk))

        outputs = output_runs(opt, gallery)
        return self._embed_run(
            opt, gallery, ids, chunks, outputs=outputs, max_usd=max_usd, reuse=reuse
        )

    def embed_queries(
        self, option_id: str, set_ids: Sequence[str], *, max_usd: float | None = None
    ) -> VectorSet:
        """Embed the query images of test sets' tune halves into one vector run.

        For sets whose queries are not gallery images: T2, T2b and T8c ask with Commons
        photos, fetched here and sent as bytes. POP query images that are not publishable are
        left out. Needs a probe of the option, like a gallery run.
        """
        ids, label = tune_query_images(self.settings, set_ids)
        return self.embed_query_images(option_id, ids, label, max_usd=max_usd)

    def embed_query_images(
        self, option_id: str, ids: Sequence[str], label: str, *, max_usd: float | None = None
    ) -> VectorSet:
        """`embed_queries` on given query image ids, into the run named for `label` (round
        R4 embeds the sealed Commons photos this way, `miroir.eval.r4`)."""
        opt = option(option_id)
        ids = self._calls.gate.store.only_publishable(ids)
        if split_commons(ids)[1] and self._commons is None:
            msg = "Commons photo(s) to embed, but no Commons fetcher was given"
            raise ValueError(msg)

        def chunks(todo: Sequence[str], skips: Counter[str]) -> Iterator[Chunk]:
            pop, commons = split_commons(todo)
            for chunk in _chunks(pop):
                yield Chunk(images=self._images(chunk))
            if commons and self._commons is not None:
                yield from _supplied_chunks(self._commons, commons, skips)

        (written,) = self._embed_run(opt, label, ids, chunks, max_usd=max_usd)
        return written

    def embed_supplied(
        self,
        option_id: str,
        label: str,
        ids: Sequence[str],
        source: SuppliesImages,
        *,
        max_usd: float | None = None,
    ) -> VectorSet:
        """Embed images whose bytes the Mac sends, into run `<option>-<label>`.

        For images the machine cannot read itself: Commons photos added to the gallery (F2),
        or the images of the records in extra/. POP ids must be publishable. Needs a probe of
        the option, like a gallery run; resumes from its checkpoint.
        """
        opt = option(option_id)
        self._calls.gate.check(ids)

        def chunks(todo: Sequence[str], skips: Counter[str]) -> Iterator[Chunk]:
            return _supplied_chunks(source, todo, skips)

        (written,) = self._embed_run(opt, label, ids, chunks, max_usd=max_usd)
        return written

    def embed_texts(
        self,
        option_id: str,
        label: str,
        ids: Sequence[str],
        texts: Sequence[str],
        *,
        max_usd: float,
    ) -> VectorSet:
        """Embed search words (no image leaves the Mac) into the run `<option>-<label>`.

        Words are tiny next to images, so no probe: each call's worst case is held against
        the budget, and the run stops before a call once it has cost `max_usd`.
        """
        opt = option(option_id)
        if opt.family is not Family.EMBEDDINGGEMMA:
            msg = f"{opt.id}: only EmbeddingGemma 2 embeds words here"
            raise ValueError(msg)
        if len(ids) != len(texts) or len(set(ids)) != len(ids) or not all(texts):
            msg = "embed_texts needs one non-empty text per id, and distinct ids"
            raise ValueError(msg)
        run = run_name(opt, label)
        out = self.settings.vectors_path(run)
        refuse_existing(out)
        start = self._clock()
        spend = self._spend(SpendKind.RUN, opt.id, run, opt.model, limit_usd=max_usd)
        done_ids: list[str] = []
        rows: list[NDArray[np.float32]] = []
        for chunk in batched(range(len(ids)), CHUNK_TEXTS):
            chunk_ids = [ids[i] for i in chunk]
            chunk_texts = [texts[i] for i in chunk]
            if spend.spent_usd >= max_usd:
                msg = f"{run}: cost ${spend.spent_usd:.4f}, at its ${max_usd:.2f} limit; stopped"
                raise BudgetError(msg)
            paid = self._timed(
                spend,
                lambda a=chunk_ids, b=chunk_texts: self._calls.embed_texts(opt, a, b),
                chunk_ids,
            )
            check_answer(run, chunk_ids, paid.result.ids, paid.result.vectors)
            if list(paid.result.ids) != chunk_ids:
                msg = f"{run}: the answer's ids do not match the words sent"
                raise ValueError(msg)
            done_ids.extend(chunk_ids)
            rows.append(paid.result.vectors)
        info = replace(
            self._info(
                opt,
                gallery=label,
                cost=spend.spent_usd,
                wall=self._clock() - start,
                skips={},
            ),
            pooling="text",
        )
        info = replace(info, notes=f"{info.notes}; words with the {TEXT_PROMPT!r} prompt")
        return write_vectors(out, done_ids, np.concatenate(rows), info)

    def probe_for(self, opt: ModelOption) -> GpuProbe | None:
        """The option's probe, else one of an option with the same model, size and resize.

        Pooling (CLS or GeM) does not change the cost, so one probe serves both.
        """
        own = load_gpu_probe(self.settings.probe_path(opt.id))
        if own is not None:
            return own
        for other in OPTIONS.values():
            same = (other.model, other.resolution, other.resize)
            if same == (opt.model, opt.resolution, opt.resize):
                found = load_gpu_probe(self.settings.probe_path(other.id))
                if found is not None:
                    logger.info("%s: using the probe of %s (same model and size)", opt.id, other.id)
                    return found
        return None

    def fill_images(self, gallery: str) -> dict[str, int]:
        """Fetch a gallery's images once into the images bucket, through the machine.

        Model runs then read images there instead of from POP. Images already there are skipped.
        Cost: the call's wall time on the machine.
        """
        ids = self._gallery(gallery)
        chunks = list(_chunks(ids))
        paid = self._hold_call(
            lambda: self._calls.fill_images([self._images(c) for c in chunks]),
            timeout_s=FILL_TIMEOUT_S * len(chunks),
            option="images",
            run=f"images-{gallery}",
            model="none",
            items=len(ids),
        )
        logger.info("images bucket: %s for %s, $%.4f", paid.result, gallery, paid.cost_usd)
        return paid.result

    # --- the one resumable run --------------------------------------------------------------

    def _embed_run(
        self,
        opt: ModelOption,
        label: str,
        wanted: Sequence[str],
        chunks: Chunker,
        *,
        outputs: dict[str, Pooling] | None = None,
        max_usd: float | None = None,
        reuse: Sequence[str] = (),
    ) -> tuple[VectorSet, ...]:
        """Probe check, limit, one paid call per chunk saved as it arrives, vectors written.

        `outputs` names the vector runs (several when one row holds several poolings); by
        default, the one run `<option>-<label>`.
        """
        run = run_name(opt, label)
        outputs = outputs or {run: opt.pooling}
        job = Resumable(
            run,
            self.settings.checkpoint_path(run),
            [self.settings.vectors_path(n) for n in outputs],
        )
        reused = {name: self._reuse(opt, name, set(wanted), job) for name in reuse}
        todo = job.todo(wanted)
        projected = check_run(
            self.budget,
            self.ledger,
            self._key_usage,
            self.probe_for(opt),
            option=opt.id,
            model=opt.model,
            resolution=opt.resolution,
            count=len(todo),
        )
        if max_usd is not None and projected > max_usd:
            msg = f"{run}: projected ${projected:.2f}, over the ${max_usd:.2f} asked for; not run"
            raise BudgetError(msg)
        limit = run_limit(
            self.budget,
            self.ledger,
            self._key_usage,
            projected,
            factor=OVERRUN_FACTOR,
            margin_usd=OVERRUN_MARGIN_USD,
        )
        logger.info(
            "%s: %d images to send (%d saved before), about $%.2f (stops past $%.2f)",
            run,
            len(todo),
            len(job.checkpoint.ids),
            projected,
            limit,
        )
        start = self._clock()
        spend = self._spend(SpendKind.RUN, opt.id, run, opt.model, limit_usd=limit)
        for chunk in chunks(todo, job.skips):
            asked = chunk.ids
            if not asked:
                continue
            paid = self._timed(
                spend, lambda c=chunk: self._calls.embed(opt, c.images, c.supplied), asked
            )
            job.keep(asked, paid.result.ids, paid.result.vectors, paid.result.skips)
            held = len(job.checkpoint.ids)
            logger.info("%s: %d/%d images, $%.4f", run, held, len(wanted), spend.spent_usd)
        # Cost of the whole run, including the sessions before a resume.
        info = self._info(
            opt,
            gallery=label,
            cost=self.ledger.total(SpendKind.RUN, run=run),
            wall=self._clock() - start,
            skips=dict(job.skips),
        )
        if reused:
            kept_from = ", ".join(f"{n} ({c})" for n, c in reused.items())
            info = replace(
                info, notes=f"{info.notes}; vectors reused from {kept_from}, not counted"
            )
        ids, vectors = job.finish(wanted)
        parts = np.split(vectors, len(outputs), axis=1)
        written = tuple(
            write_vectors(
                self.settings.vectors_path(name),
                ids,
                part,
                _one_pooling(info, pooling, shared=len(outputs) > 1),
            )
            for (name, pooling), part in zip(outputs.items(), parts, strict=True)
        )
        job.checkpoint.remove()
        return written

    def _reuse(self, opt: ModelOption, name: str, wanted: set[str], job: Resumable) -> int:
        """Copy the vectors of an earlier run into the checkpoint. Returns how many were taken.

        Refused unless the run used the same model, size, pooling and number type on a GPU:
        on Modal before 2026-11-03 (notes "Modal L4, float32") or on Scaleway since.
        """
        done = read_vectors(self.settings.vectors_path(name))
        info = done.info
        same = (info.model, info.resolution, info.pooling) == (
            opt.model,
            opt.resolution,
            opt.pooling.value,
        ) and _made_on_a_gpu(info.notes, opt)
        if not same:
            msg = f"{name} was not embedded with the settings of {opt.id}; not reused"
            raise ValueError(msg)
        held = set(job.checkpoint.ids)
        rows = [n for n, i in enumerate(done.ids) if i in wanted and i not in held]
        for block in batched(rows, CHUNK_IMAGES * 10):
            picked = [done.ids[n] for n in block]
            job.checkpoint.append(picked, np.asarray(done.vectors[list(block)], dtype=np.float16))
        logger.info("%s: reused %d vectors from %s", opt.id, len(rows), name)
        return len(rows)

    # --- helpers ----------------------------------------------------------------------------

    def _spend(
        self, kind: SpendKind, option_id: str, run: str, model: str, *, limit_usd: float
    ) -> RunSpend:
        return RunSpend(
            self.ledger,
            self.budget,
            self._key_usage,
            limit_usd=limit_usd,
            per_item_usd=0.0,  # a call's cost is its measured wall time
            kind=kind,
            option=option_id,
            run=run,
            model=model,
            service=self._service,
        )

    def _timed[T](self, spend: RunSpend, call: Callable[[], T], items: Sized) -> Paid[T]:
        worst = self._calls.machine.usd_per_s * CHUNK_TIMEOUT_S
        hold = spend.call(len(items), worst_usd=worst)
        return timed(hold, call, self._calls.machine, self._clock)

    def _hold_call[T](
        self,
        call: Callable[[], T],
        *,
        timeout_s: int,
        option: str,
        run: str,
        model: str,
        items: int,
    ) -> Paid[T]:
        return paid_call(
            PaidCall(self.ledger, self.budget, self._key_usage, self._clock, self._service),
            call,
            machine=self._calls.machine,
            timeout_s=timeout_s,
            kind=SpendKind.RUN,
            option=option,
            run=run,
            model=model,
            items=items,
        )

    def _gallery(self, name: str) -> list[str]:
        path = self.settings.gallery_path(name)
        if not path.exists():
            msg = f"no gallery {name!r} at {path}: run `miroir testsets build` first"
            raise FileNotFoundError(msg)
        ids = read_gallery(self.settings, name)
        self._calls.gate.check(ids)  # the whole gallery, before paying for any of it
        return ids

    def _images(self, ids: Sequence[str]) -> list[Image]:
        self._calls.gate.check(ids)
        images = self._calls.gate.store.images
        return [images[i] for i in ids]

    def _info(
        self,
        opt: ModelOption,
        *,
        gallery: str,
        cost: float,
        wall: float,
        skips: dict[str, int],
    ) -> RunInfo:
        notes = f"{self._service.value.capitalize()} {self._calls.machine.gpu}, {opt.dtype}"
        if opt.family is Family.DINO:
            notes += f", {opt.resize}"
        if skips:
            notes += f"; skipped: {skips}"
        return RunInfo(
            option=opt.experiment,
            model=opt.model,
            resolution=opt.resolution,
            pooling=opt.pooling.value,
            gallery=gallery,
            created=_today(),
            cost_usd=cost,
            wall_time_s=wall,
            route=Route.HOSTED,
            notes=notes,
        )


def _supplied_chunks(
    source: SuppliesImages, ids: Sequence[str], skips: Counter[str]
) -> Iterator[Chunk]:
    """Images fetched on the Mac and sent as bytes, one call per chunk."""
    for chunk in _chunks(ids):
        supplied = [
            SuppliedImage(f.image_id, f.jpeg)
            for f in source.fetch_all(chunk)
            if kept(f, skips) and f.jpeg is not None
        ]
        yield Chunk(supplied=supplied)


def _made_on_a_gpu(notes: str, opt: ModelOption) -> bool:
    """Whether run notes start "Modal <gpu>, <dtype>" or "Scaleway <gpu>, <dtype>"."""
    return re.match(rf"(Modal|Scaleway) [^,;]+, {re.escape(opt.dtype)}\b", notes) is not None


def _one_pooling(info: RunInfo, pooling: Pooling, *, shared: bool) -> RunInfo:
    if not shared:
        return info
    notes = f"{info.notes}; CLS and GeM from one pass: cost and time cover both runs"
    return replace(info, pooling=pooling.value, notes=notes)


def _chunks(ids: Sequence[str]) -> Iterator[list[str]]:
    for chunk in batched(ids, CHUNK_IMAGES):
        yield list(chunk)


def _today() -> date:
    return datetime.now(UTC).date()
