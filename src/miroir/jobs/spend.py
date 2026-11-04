"""Money rules for paid model work (experiments.md §1 rule 5).

- Every paid call is appended to one ledger (`Settings.spend_ledger`) as soon as it returns,
  fails or is interrupted, tagged with the service that billed it.
- Total spend is the larger of the OpenRouter lines and what OpenRouter says the key has
  spent, plus the Modal lines (runs before 2026-11-03). Scaleway lines are left out: the
  machines have their own cap (`miroir.jobs.scaleway`).
- Every paid call is held against the budget while in flight (`SpendLedger.hold`): under a
  lock shared by every process on the Mac, its worst case is written as a hold line, and the
  call is refused if spend plus every open hold plus it would pass the usable budget
  (`budget_usd` minus `budget_reserve_usd`, caps in `miroir.config`). The line of its real
  cost settles the hold. Two runs started side by side therefore cannot both spend the last
  dollars. A call that fails after it may have been billed is settled at its expected cost,
  marked `estimated`.
- A probe embeds 100 images and stores the cost per image (`Settings.probe_path`).
  Probes together, on both services, may not spend more than `probe_budget_usd`.
- A full run needs a probe on file for the same model and image size, and refuses to start
  if spend plus its projected cost would pass the usable budget. It stops past its own limit
  (`run_limit`), which never passes what the usable budget has left.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import ConfigDict, TypeAdapter, ValidationError

from miroir.config import Settings
from miroir.jobs.durable import append_lines, locked, read_lines

logger = logging.getLogger(__name__)

PROBE_IMAGES = 100
#: A paid call is held at this many times its expected cost while in flight.
CALL_SLACK = 2.0


class SpendKind(StrEnum):
    PROBE = "probe"
    RUN = "run"
    CHECK = "check"
    """One-off calls to learn how the API behaves."""


class Service(StrEnum):
    """Who billed a call."""

    OPENROUTER = "openrouter"
    MODAL = "modal"
    """Calls on Modal GPUs, before 2026-11-03. Kept so old ledger lines still read."""
    SCALEWAY = "scaleway"
    """Calls on a Scaleway GPU machine, paid from the ministry's project, not from this budget.
    Machines have their own cap and ledger (`miroir.jobs.scaleway`)."""


class BudgetError(RuntimeError):
    """A paid call was refused because it could break the spending cap."""


@dataclass(frozen=True, slots=True)
class SpendLine:
    __pydantic_config__ = ConfigDict(extra="forbid")

    at: datetime
    kind: SpendKind
    option: str
    run: str
    model: str
    cost_usd: float
    items: int
    service: Service = Service.OPENROUTER
    """Lines written before a second service arrived carry no service: they are OpenRouter's."""
    hold_usd: float = 0.0
    """On a hold line: the worst case of a call in flight. Its `cost_usd` is 0."""
    hold: str = ""
    """Ties a hold line to the line that settles it."""
    pid: int = 0
    """The process that holds: a hold of a process that has stopped no longer counts."""
    estimated: bool = False
    """The call failed after it may have been billed: `cost_usd` is its expected cost."""


class Projection(Protocol):
    """A probe as `check_run` reads it: what it measured and what a run would cost."""

    @property
    def model(self) -> str: ...
    @property
    def resolution(self) -> int: ...
    def projected_usd(self, count: int) -> float: ...


@dataclass(frozen=True, slots=True)
class Probe:
    """Cost of embedding `images` images with one option, measured on OpenRouter."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    option: str
    model: str
    resolution: int
    gallery: str
    images: int
    cost_usd: float
    created: date

    @property
    def usd_per_image(self) -> float:
        return self.cost_usd / self.images

    def projected_usd(self, count: int) -> float:
        return self.usd_per_image * count


_LINE = TypeAdapter(SpendLine)
_PROBE = TypeAdapter(Probe)


@dataclass(frozen=True, slots=True)
class Budget:
    cap_usd: float
    reserve_usd: float
    probe_cap_usd: float

    @classmethod
    def from_settings(cls, settings: Settings) -> Budget:
        return cls(
            cap_usd=settings.budget_usd,
            reserve_usd=settings.budget_reserve_usd,
            probe_cap_usd=settings.probe_budget_usd,
        )

    @property
    def usable_usd(self) -> float:
        return self.cap_usd - self.reserve_usd


class Bill:
    """What a held call cost: the block of `SpendLedger.hold` sets `cost_usd` once known."""

    __slots__ = ("cost_usd",)

    def __init__(self) -> None:
        self.cost_usd: float | None = None


class SpendLedger:
    """Append-only record of what paid calls cost. Safe to share between threads and processes."""

    def __init__(self, path: Path, *, alive: Callable[[int], bool] | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._alive = alive or _alive

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        kind: SpendKind,
        *,
        option: str,
        run: str,
        model: str,
        cost_usd: float,
        items: int,
        service: Service = Service.OPENROUTER,
        estimated: bool = False,
    ) -> None:
        line = SpendLine(
            datetime.now(UTC),
            kind,
            option,
            run,
            model,
            cost_usd,
            items,
            service,
            estimated=estimated,
        )
        with self._locked():
            self._append(line)

    def lines(self) -> list[SpendLine]:
        try:
            return [_LINE.validate_json(line) for line in read_lines(self._path)]
        except ValidationError as err:
            msg = f"{self._path} is damaged; fix it by hand before spending more: {err}"
            raise BudgetError(msg) from err

    def total(
        self,
        kind: SpendKind | None = None,
        service: Service | None = None,
        *,
        run: str | None = None,
    ) -> float:
        return sum(
            line.cost_usd
            for line in self.lines()
            if (kind is None or line.kind == kind)
            and (service is None or line.service == service)
            and (run is None or line.run == run)
        )

    def held_usd(self, lines: Sequence[SpendLine] | None = None) -> float:
        """Worst cases of the calls in flight now, in running processes, Scaleway left out."""
        lines = self.lines() if lines is None else lines
        settled = {line.hold for line in lines if line.hold and not line.hold_usd}
        held = 0.0
        for line in lines:
            if not line.hold_usd or line.hold in settled or line.service is Service.SCALEWAY:
                continue
            if not self._alive(line.pid):
                logger.debug("%s: hold %s of a stopped process left out", self._path, line.hold)
                continue
            held += line.hold_usd
        return held

    @contextmanager
    def hold(
        self,
        budget: Budget,
        key_usage_usd: float | None,
        *,
        worst_usd: float,
        expected_usd: float,
        kind: SpendKind,
        option: str,
        run: str,
        model: str,
        items: int,
        service: Service = Service.OPENROUTER,
    ) -> Generator[Bill]:
        """Hold `worst_usd` for one paid call while the block runs, then settle its cost.

        Refused, before anything is sent, if spend plus the open holds plus `worst_usd` would
        pass the usable budget. The block sets `bill.cost_usd`. If it raises first, the call is
        settled at `expected_usd`, marked estimated, unless the service answered with an error
        status, which bills nothing.
        """
        hold_id = uuid.uuid4().hex
        with self._locked():
            committed = committed_usd(self, key_usage_usd)
            if committed + worst_usd > budget.usable_usd:
                msg = (
                    f"{run}: a call may cost up to ${worst_usd:.4f}, and ${committed:.2f} is "
                    "spent or held by calls in flight; together that passes the "
                    f"${budget.usable_usd:.2f} usable budget"
                )
                raise BudgetError(msg)
            self._append(
                SpendLine(
                    datetime.now(UTC),
                    kind,
                    option,
                    run,
                    model,
                    0.0,
                    0,
                    service,
                    hold_usd=worst_usd,
                    hold=hold_id,
                    pid=os.getpid(),
                )
            )
        bill = Bill()
        cost, estimated = expected_usd, True
        try:
            yield bill
        except BaseException as err:
            if bill.cost_usd is None and not billed(err):
                cost, estimated = 0.0, False
            raise
        finally:
            if bill.cost_usd is not None:
                cost, estimated = bill.cost_usd, False
            settle = SpendLine(
                datetime.now(UTC),
                kind,
                option,
                run,
                model,
                cost,
                items,
                service,
                hold=hold_id,
                pid=os.getpid(),
                estimated=estimated,
            )
            with self._locked():
                self._append(settle)
            bill.cost_usd = cost

    @contextmanager
    def _locked(self) -> Generator[None]:
        with self._lock, locked(self._path):
            yield

    def _append(self, line: SpendLine) -> None:
        append_lines(self._path, [_LINE.dump_json(line)])


def billed(err: BaseException) -> bool:
    """Whether a failed call may have been billed: anything but an answered error status."""
    from miroir.jobs.openrouter import OpenRouterError  # noqa: PLC0415 (no import cycle at load)

    return not (isinstance(err, OpenRouterError) and err.status is not None)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def spent_usd(ledger: SpendLedger, key_usage_usd: float | None) -> float:
    """Total spent so far, on every service.

    For OpenRouter, the ledger or the key's own count, whichever is higher; for the rest,
    the ledger. Scaleway lines are left out: another account pays them.
    """
    return _spent(ledger.lines(), key_usage_usd)


def committed_usd(ledger: SpendLedger, key_usage_usd: float | None) -> float:
    """Spent so far plus the worst cases of the calls in flight (the ledger read once)."""
    lines = ledger.lines()
    return _spent(lines, key_usage_usd) + ledger.held_usd(lines)


def _spent(lines: Sequence[SpendLine], key_usage_usd: float | None) -> float:
    openrouter = sum(line.cost_usd for line in lines if line.service is Service.OPENROUTER)
    if key_usage_usd is not None:
        openrouter = max(openrouter, key_usage_usd)
    others = sum(
        line.cost_usd
        for line in lines
        if line.service not in (Service.OPENROUTER, Service.SCALEWAY)
    )
    return openrouter + others


def check_headroom(
    budget: Budget, ledger: SpendLedger, key_usage_usd: float | None, max_usd: float
) -> None:
    """Refuse to start a run that may spend up to `max_usd` when that could pass the usable
    budget (a run with no probe, which stops itself at `max_usd`)."""
    spent = spent_usd(ledger, key_usage_usd)
    if spent + max_usd > budget.usable_usd:
        msg = (
            f"spent ${spent:.2f}; ${max_usd:.2f} more would pass the ${budget.usable_usd:.2f} "
            "usable budget. Ask the project lead."
        )
        raise BudgetError(msg)


def check_probe(budget: Budget, ledger: SpendLedger, key_usage_usd: float | None) -> float:
    """Refuse a probe once probes have used their cap. Returns what a probe may still spend."""
    left = budget.probe_cap_usd - ledger.total(SpendKind.PROBE)
    if left <= 0:
        msg = f"probes have spent their ${budget.probe_cap_usd:.2f} cap; raise it to probe again"
        raise BudgetError(msg)
    room = budget.usable_usd - committed_usd(ledger, key_usage_usd)
    if room <= 0:
        msg = f"total spend has reached the ${budget.usable_usd:.2f} usable budget"
        raise BudgetError(msg)
    return min(left, room)


def check_run(
    budget: Budget,
    ledger: SpendLedger,
    key_usage_usd: float | None,
    probe: Projection | None,
    *,
    option: str,
    model: str,
    resolution: int,
    count: int,
) -> float:
    """Refuse a full run without a matching probe, or one that would pass the cap.

    Returns the projected cost in dollars.
    """
    if probe is None:
        msg = f"no probe on file for {option}: run it with --probe first"
        raise BudgetError(msg)
    if probe.model != model or probe.resolution != resolution:
        msg = (
            f"the {option} probe measured {probe.model} at {probe.resolution} px, "
            f"not {model} at {resolution} px: probe again"
        )
        raise BudgetError(msg)
    projected = probe.projected_usd(count)
    spent = committed_usd(ledger, key_usage_usd)
    if spent + projected > budget.usable_usd:
        msg = (
            f"{option} on {count} images would cost about ${projected:.2f}. Spent so far "
            f"(calls in flight included): ${spent:.2f}. Together that passes the "
            f"${budget.usable_usd:.2f} usable budget (${budget.cap_usd:.2f} cap minus "
            f"${budget.reserve_usd:.2f} reserve). "
            "Raising the cap (BUDGET_USD in config.py) needs the project lead's yes."
        )
        raise BudgetError(msg)
    return projected


def run_limit(
    budget: Budget,
    ledger: SpendLedger,
    key_usage_usd: float | None,
    projected_usd: float,
    *,
    factor: float,
    margin_usd: float = 0.0,
    max_usd: float | None = None,
) -> float:
    """Where a run stops: `factor` times its projection plus `margin_usd`, at most `max_usd`,
    and never past what the usable budget has left (calls in flight counted)."""
    room = budget.usable_usd - committed_usd(ledger, key_usage_usd)
    limit = min(projected_usd * factor + margin_usd, room)
    return limit if max_usd is None else min(limit, max_usd)


class RunSpend:
    """One run's paid calls: each held against the budget; the run stops past its limit.

    `per_item_usd` is the expected cost of one item (from a probe); a call is held at
    `CALL_SLACK` times its expected cost unless it names its own worst case. Calls may run
    in parallel threads.
    """

    def __init__(
        self,
        ledger: SpendLedger,
        budget: Budget,
        key_usage_usd: float | None,
        *,
        limit_usd: float,
        per_item_usd: float,
        kind: SpendKind,
        option: str,
        run: str,
        model: str,
        service: Service = Service.OPENROUTER,
    ) -> None:
        self.ledger = ledger
        self.budget = budget
        self.limit_usd = limit_usd
        self._key_usage = key_usage_usd
        self._per_item = per_item_usd
        self.kind = kind
        self._option = option
        self.run = run
        self._model = model
        self._service = service
        self._lock = threading.Lock()
        self.spent_usd = 0.0
        """What this run's calls have cost in this session."""

    @contextmanager
    def call(
        self, items: int, *, worst_usd: float | None = None, model: str | None = None
    ) -> Generator[Bill]:
        """Hold one call of `items` items. Refused once the run has passed its limit."""
        with self._lock:
            if self.spent_usd > self.limit_usd:
                msg = (
                    f"{self.run}: spent ${self.spent_usd:.4f}, over its ${self.limit_usd:.4f} "
                    "limit; stopped"
                )
                raise BudgetError(msg)
        expected = self._per_item * items
        held: list[Bill] = []
        try:
            with self.ledger.hold(
                self.budget,
                self._key_usage,
                worst_usd=expected * CALL_SLACK if worst_usd is None else worst_usd,
                expected_usd=expected,
                kind=self.kind,
                option=self._option,
                run=self.run,
                model=model or self._model,
                items=items,
                service=self._service,
            ) as bill:
                held.append(bill)
                yield bill
        finally:
            # Once the hold is settled, the bill holds what was written to the ledger.
            with self._lock:
                self.spent_usd += sum(b.cost_usd or 0.0 for b in held)


def save_probe(path: Path, probe: Probe) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PROBE.dump_json(probe, indent=2) + b"\n")


def load_probe(path: Path) -> Probe | None:
    if not path.exists():
        return None
    try:
        return _PROBE.validate_json(path.read_bytes())
    except (ValidationError, json.JSONDecodeError) as err:
        msg = f"{path} is not a probe result: {err}"
        raise BudgetError(msg) from err
