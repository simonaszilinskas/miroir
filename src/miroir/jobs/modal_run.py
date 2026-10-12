"""Embedding runs on Modal, driven from the Mac: probe, gallery run, and R0 on AmsterTime.

Money (experiments.md §1 rule 5). Modal bills by the second for GPU, CPU and memory. Each
remote call is timed on the Mac, from the call to its answer, and costs that many seconds at
the function's published price (`modal_spec.Machine.usd_per_s`): a little more than Modal
bills, since the clock also runs while the call travels. The cost goes into the one project
ledger, tagged `modal`, even when the call fails. Before every call, total spend plus the
call's worst case (its timeout at full price) must stay inside the usable budget. A gallery
run needs a 100-image probe of the same model and size on file, and stops once it costs
`OVERRUN_FACTOR` times its projection. Each answer is saved to a checkpoint as it arrives,
so a stopped run resumes without paying twice.

Privacy (technical spec §9): Modal is an outside service, so only images of `publishable`
records go there; a gallery with any other image is refused.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from itertools import batched
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter, ValidationError

from miroir.config import SEED, Settings
from miroir.eval import amstertime
from miroir.eval.rerank import MatchChunk, MatchTask, append_matches, read_matches
from miroir.eval.sets import read_gallery
from miroir.eval.vectors import Route, RunInfo, VectorSet, write_vectors
from miroir.jobs.checkpoint import Checkpoint
from miroir.jobs.modal_spec import (
    CHUNK_IMAGES,
    CHUNK_TIMEOUT_S,
    DOWNLOAD_MACHINE,
    DOWNLOAD_TIMEOUT_S,
    FILL_CONTAINERS,
    FILL_MACHINE,
    FILL_TIMEOUT_S,
    L4,
    MATCH_QUERIES,
    OPTIONS,
    R0_MACHINE,
    R0_OPTIONS,
    R0_TIMEOUT_S,
    ChunkResult,
    Family,
    Machine,
    ModalOption,
    Pooling,
    option,
)
from miroir.jobs.privacy import PublishableImages
from miroir.jobs.spend import (
    PROBE_IMAGES,
    Budget,
    BudgetError,
    Service,
    SpendKind,
    SpendLedger,
    check_probe,
    check_run,
    spent_usd,
)
from miroir.records.model import Image

logger = logging.getLogger(__name__)

#: A run stops once it costs this many times its projection, plus the margin below.
OVERRUN_FACTOR = 1.5
#: Start-up varies more than per-image time; small runs get this much slack on top.
OVERRUN_MARGIN_USD = 0.25
SERVICE_NAME = "Modal"


class RemoteCalls(Protocol):
    """The calls a run makes. `miroir.jobs.modal_app.ModalCalls` sends them to Modal."""

    def embed(self, option: ModalOption, images: Sequence[Image]) -> ChunkResult: ...
    def fill_images(self, chunks: Sequence[Sequence[Image]]) -> dict[str, int]: ...
    def amstertime_download(self) -> int: ...
    def amstertime_embed(self, option: ModalOption) -> ChunkResult: ...
    def match(self, tasks: Sequence[MatchTask]) -> MatchChunk: ...


@dataclass(frozen=True, slots=True)
class ModalProbe:
    """Time and cost of embedding about 100 gallery images with one option on Modal."""

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


_PROBE = TypeAdapter(ModalProbe)


def save_modal_probe(path: Path, probe: ModalProbe) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PROBE.dump_json(probe, indent=2) + b"\n")


def load_modal_probe(path: Path) -> ModalProbe | None:
    if not path.exists():
        return None
    try:
        return _PROBE.validate_json(path.read_bytes())
    except (ValidationError, json.JSONDecodeError) as err:
        msg = f"{path} is not a Modal probe: {err}"
        raise BudgetError(msg) from err


def run_name(opt: ModalOption, gallery: str) -> str:
    return f"{opt.id}-{gallery}"


def output_runs(opt: ModalOption, gallery: str) -> dict[str, Pooling]:
    """The vector runs a gallery run writes, with their pooling. CLS+GeM writes two."""
    if opt.pooling is Pooling.CLS_GEM:
        return {f"{opt.id}-{p.value}-{gallery}": p for p in (Pooling.CLS, Pooling.GEM)}
    return {run_name(opt, gallery): opt.pooling}


@dataclass(frozen=True, slots=True)
class _Paid[T]:
    result: T
    cost_usd: float
    wall_s: float


class ModalRunner:
    """Runs embedding jobs on Modal and logs what each call costs."""

    def __init__(
        self,
        settings: Settings,
        calls: RemoteCalls,
        *,
        key_usage_usd: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._calls = calls
        self._key_usage = key_usage_usd
        """What OpenRouter says its key has spent, read once by the caller (or None)."""
        self._clock = clock
        self.ledger = SpendLedger(settings.spend_ledger)
        self.budget = Budget.from_settings(settings)
        self._loaded: PublishableImages | None = None

    # --- the three entry points -------------------------------------------------------------

    def probe(self, option_id: str, gallery: str) -> tuple[ModalProbe, int]:
        """Embed 100 images of `gallery`, store time and cost. Returns it and the gallery size."""
        opt = option(option_id)
        ids = self._gallery(gallery)
        left = check_probe(self.budget, self.ledger, self._key_usage)
        worst = opt.machine.usd_per_s * CHUNK_TIMEOUT_S
        if worst > left:
            msg = f"a probe may cost up to ${worst:.2f}; probes have ${left:.2f} left"
            raise BudgetError(msg)
        sample = random.Random(SEED).sample(ids, min(PROBE_IMAGES, len(ids)))
        run = f"probe-{run_name(opt, gallery)}"
        paid = self._paid(
            lambda: self._calls.embed(opt, self._images(sample)),
            machine=opt.machine,
            timeout_s=CHUNK_TIMEOUT_S,
            kind=SpendKind.PROBE,
            option=opt.id,
            run=run,
            model=opt.model,
            items=len(sample),
        )
        done = paid.result
        if not done.ids:
            msg = f"probe for {opt.id}: no image could be embedded ({done.skips})"
            raise RuntimeError(msg)
        _check_chunk(done, opt)
        result = ModalProbe(
            option=opt.id,
            model=opt.model,
            resolution=opt.resolution,
            gallery=gallery,
            images=len(done.ids),
            fixed_s=max(paid.wall_s - done.embed_s, 0.0),
            per_image_s=done.embed_s / len(done.ids),
            usd_per_s=opt.machine.usd_per_s,
            cost_usd=paid.cost_usd,
            created=_today(),
        )
        save_modal_probe(self.settings.probe_path(opt.id), result)
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
        self, option_id: str, gallery: str, *, max_usd: float | None = None
    ) -> tuple[VectorSet, ...]:
        """Embed every image of a gallery and write its vector runs (technical spec §7.1).

        One run, or two for CLS+GeM (`output_runs`). `max_usd` refuses the run when the probe
        projects a higher cost.
        """
        opt = option(option_id)
        run = run_name(opt, gallery)
        outputs = output_runs(opt, gallery)
        for name in outputs:
            _refuse_existing(self.settings.vectors_path(name))
        ids = self._gallery(gallery)
        checkpoint = Checkpoint(self.settings.checkpoint_path(run))
        held = set(checkpoint.ids)
        todo = [i for i in ids if i not in held]
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
        limit = projected * OVERRUN_FACTOR + OVERRUN_MARGIN_USD
        logger.info(
            "%s: %d images to send (%d saved before), about $%.2f (stops past $%.2f)",
            run,
            len(todo),
            len(held),
            projected,
            limit,
        )
        start = self._clock()
        skips: dict[str, int] = {}
        cost = 0.0
        for chunk in _chunks(todo):
            paid = self._paid(
                lambda chunk=chunk: self._calls.embed(opt, self._images(chunk)),
                machine=opt.machine,
                timeout_s=CHUNK_TIMEOUT_S,
                kind=SpendKind.RUN,
                option=opt.id,
                run=run,
                model=opt.model,
                items=len(chunk),
            )
            cost += paid.cost_usd
            _check_chunk(paid.result, opt)
            if paid.result.ids:
                checkpoint.append(list(paid.result.ids), _unit_float16(paid.result.vectors))
            for reason, n in paid.result.skips.items():
                skips[reason] = skips.get(reason, 0) + n
            logger.info("%s: %d/%d images, $%.4f", run, len(checkpoint.ids), len(ids), cost)
            if cost > limit:
                msg = f"{run}: cost ${cost:.4f}, over its ${limit:.4f} limit; stopped"
                raise BudgetError(msg)
        # Cost of the whole run, including the sessions before a resume.
        total = self.ledger.total(SpendKind.RUN, run=run)
        info = self._info(
            opt,
            gallery=gallery,
            machine=opt.machine,
            cost=total,
            wall=self._clock() - start,
            skips=skips,
        )
        vectors = checkpoint.vectors()
        parts = np.split(vectors, len(outputs), axis=1)
        written = tuple(
            write_vectors(
                self.settings.vectors_path(name),
                checkpoint.ids,
                part,
                _one_pooling(info, pooling, shared=len(outputs) > 1),
            )
            for (name, pooling), part in zip(outputs.items(), parts, strict=True)
        )
        checkpoint.remove()
        return written

    def probe_for(self, opt: ModalOption) -> ModalProbe | None:
        """The option's probe, else one of an option with the same model, size and resize.

        Pooling (CLS or GeM) does not change the cost, so one probe serves both.
        """
        own = load_modal_probe(self.settings.probe_path(opt.id))
        if own is not None:
            return own
        for other in OPTIONS.values():
            same = (other.model, other.resolution, other.resize)
            if same == (opt.model, opt.resolution, opt.resize):
                found = load_modal_probe(self.settings.probe_path(other.id))
                if found is not None:
                    logger.info("%s: using the probe of %s (same model and size)", opt.id, other.id)
                    return found
        return None

    def fill_images(self, gallery: str) -> dict[str, int]:
        """Fetch a gallery's images once into the Modal image volume, several containers at once.

        Model runs then read images there instead of from POP. Images already there are skipped.
        Cost: the call's wall time for every container that may run.
        """
        ids = self._gallery(gallery)
        machine = Machine(
            FILL_MACHINE.gpu,
            FILL_MACHINE.cpu * FILL_CONTAINERS,
            FILL_MACHINE.memory_gib * FILL_CONTAINERS,
        )
        chunks = list(_chunks(ids))
        paid = self._paid(
            lambda: self._calls.fill_images([self._images(c) for c in chunks]),
            machine=machine,
            timeout_s=FILL_TIMEOUT_S * len(chunks) // FILL_CONTAINERS + FILL_TIMEOUT_S,
            kind=SpendKind.RUN,
            option="images",
            run=f"images-{gallery}",
            model="none",
            items=len(ids),
        )
        logger.info("image volume: %s for %s, $%.4f", paid.result, gallery, paid.cost_usd)
        return paid.result

    def match(self, name: str, tasks: Sequence[MatchTask], *, max_usd: float) -> int:
        """Keypoint matches of each query against its shortlist (step E1), on a Modal GPU.

        Each answer is appended to `Settings.matches_path(name)` as it arrives, so a stopped
        run resumes. Stops before a call once the run has cost `max_usd`. Returns queries done.
        """
        path = self.settings.matches_path(name)
        held = read_matches(path)
        todo = [t for t in tasks if t.query_id not in held]
        self._images(sorted({i for t in todo for i in (t.image_id, *t.candidates)}))  # privacy
        logger.info("%s: %d queries to match (%d saved before)", name, len(todo), len(held))
        cost = 0.0
        missing: set[str] = set()
        for chunk in batched(todo, MATCH_QUERIES):
            if cost >= max_usd:
                msg = f"{name}: cost ${cost:.4f}, at its ${max_usd:.2f} limit; stopped"
                raise BudgetError(msg)
            paid = self._paid(
                lambda chunk=chunk: self._calls.match(chunk),
                machine=L4,
                timeout_s=CHUNK_TIMEOUT_S,
                kind=SpendKind.RUN,
                option="E1",
                run=name,
                model="aliked+lightglue",
                items=len(chunk),
            )
            cost += paid.cost_usd
            append_matches(path, paid.result.queries)
            missing.update(paid.result.missing)
            held.update({q.query_id: q for q in paid.result.queries})
            logger.info("%s: %d/%d queries, $%.4f", name, len(held), len(tasks), cost)
        if missing:
            logger.warning("%s: %d image(s) could not be read on Modal", name, len(missing))
        return len(held)

    def r0(self, name: str) -> VectorSet:
        """Round R0: download AmsterTime into a Modal volume, embed all 2,462 images there."""
        if name not in R0_OPTIONS:
            msg = f"unknown R0 model {name!r}; known: {', '.join(R0_OPTIONS)}"
            raise ValueError(msg)
        opt = R0_OPTIONS[name]
        run = run_name(opt, amstertime.GALLERY)
        out = self.settings.vectors_path(run)
        _refuse_existing(out)
        start = self._clock()
        downloaded = self._paid(
            self._calls.amstertime_download,
            machine=DOWNLOAD_MACHINE,
            timeout_s=DOWNLOAD_TIMEOUT_S,
            kind=SpendKind.RUN,
            option="R0",
            run=f"{run}-download",
            model="none",
            items=0,
        )
        logger.info("AmsterTime on Modal: %d images", downloaded.result)
        paid = self._paid(
            lambda: self._calls.amstertime_embed(opt),
            machine=R0_MACHINE,
            timeout_s=R0_TIMEOUT_S,
            kind=SpendKind.RUN,
            option="R0",
            run=run,
            model=opt.model,
            items=downloaded.result,
        )
        done = paid.result
        _check_chunk(done, opt)
        amstertime.check_ids(done.ids)
        cost = downloaded.cost_usd + paid.cost_usd
        info = self._info(
            opt,
            gallery=amstertime.GALLERY,
            machine=R0_MACHINE,
            cost=cost,
            wall=self._clock() - start,
            skips=done.skips,
        )
        logger.info(
            "%s: $%.4f, model load %.0f s, embed %.0f s", run, cost, done.load_s, done.embed_s
        )
        return write_vectors(out, done.ids, done.vectors, info)

    # --- helpers ----------------------------------------------------------------------------

    def _paid[T](
        self,
        call: Callable[[], T],
        *,
        machine: Machine,
        timeout_s: int,
        kind: SpendKind,
        option: str,
        run: str,
        model: str,
        items: int,
    ) -> _Paid[T]:
        """Make one remote call if its worst case fits the budget; log its cost, even on failure."""
        worst = machine.usd_per_s * timeout_s
        spent = spent_usd(self.ledger, self._key_usage)
        if spent + worst > self.budget.usable_usd:
            msg = (
                f"{run}: a call may cost up to ${worst:.2f} and ${spent:.2f} is spent; together "
                f"that passes the ${self.budget.usable_usd:.2f} usable budget"
            )
            raise BudgetError(msg)
        start = self._clock()
        try:
            result = call()
        finally:
            wall = self._clock() - start
            self.ledger.record(
                kind,
                option=option,
                run=run,
                model=model,
                cost_usd=wall * machine.usd_per_s,
                items=items,
                service=Service.MODAL,
            )
        return _Paid(result, wall * machine.usd_per_s, wall)

    def _gallery(self, name: str) -> list[str]:
        path = self.settings.gallery_path(name)
        if not path.exists():
            msg = f"no gallery {name!r} at {path}: run `miroir testsets build` first"
            raise FileNotFoundError(msg)
        ids = read_gallery(self.settings, name)
        self._store().check(ids, SERVICE_NAME)
        return ids

    def _images(self, ids: Sequence[str]) -> list[Image]:
        store = self._store()
        store.check(ids, SERVICE_NAME)
        return [store.images[i] for i in ids]

    def _store(self) -> PublishableImages:
        if self._loaded is None:
            self._loaded = PublishableImages.load(self.settings)
        return self._loaded

    def _info(
        self,
        opt: ModalOption,
        *,
        gallery: str,
        machine: Machine,
        cost: float,
        wall: float,
        skips: dict[str, int],
    ) -> RunInfo:
        notes = f"Modal {machine.gpu}, {opt.dtype}"
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


def _one_pooling(info: RunInfo, pooling: Pooling, *, shared: bool) -> RunInfo:
    if not shared:
        return info
    notes = f"{info.notes}; CLS and GeM from one pass: cost and time cover both runs"
    return replace(info, pooling=pooling.value, notes=notes)


def _chunks(ids: Sequence[str]) -> Iterator[list[str]]:
    for chunk in batched(ids, CHUNK_IMAGES):
        yield list(chunk)


def _check_chunk(result: ChunkResult, opt: ModalOption) -> None:
    rows = result.vectors.shape[0] if result.vectors.ndim == 2 else -1
    if rows != len(result.ids):
        msg = f"{opt.id}: {len(result.ids)} ids but vectors of shape {result.vectors.shape}"
        raise ValueError(msg)


def _unit_float16(vectors: NDArray[np.float32]) -> NDArray[np.float16]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float16)


def _refuse_existing(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        msg = f"{out} already exists; a finished run is never overwritten"
        raise FileExistsError(msg)


def _today() -> date:
    return datetime.now(UTC).date()
