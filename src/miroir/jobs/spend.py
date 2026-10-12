"""Money rules for paid model work, OpenRouter and Modal together (experiments.md §1 rule 5).

- Every paid call is appended to one ledger (`Settings.spend_ledger`) as soon as it returns,
  tagged with the service that billed it.
- Total spend is the Modal lines plus the larger of the OpenRouter lines and what OpenRouter
  says the key has spent.
- A probe embeds 100 images and stores the cost per image (`Settings.probe_path`).
  Probes together, on both services, may not spend more than `probe_budget_usd`.
- A full run needs a probe on file for the same model and image size, and refuses to start
  if total spend plus its projected cost would pass `openrouter_budget_usd` minus
  `openrouter_reserve_usd`.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import ConfigDict, TypeAdapter, ValidationError

from miroir.config import Settings

logger = logging.getLogger(__name__)

PROBE_IMAGES = 100


class SpendKind(StrEnum):
    PROBE = "probe"
    RUN = "run"
    CHECK = "check"
    """One-off calls to learn how the API behaves."""


class Service(StrEnum):
    """Who billed a call."""

    OPENROUTER = "openrouter"
    MODAL = "modal"


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
    """Lines written before Modal arrived carry no service: they are OpenRouter's."""


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


class SpendLedger:
    """Append-only record of what paid calls cost. Safe to use from several threads."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

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
    ) -> None:
        line = SpendLine(datetime.now(UTC), kind, option, run, model, cost_usd, items, service)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("ab") as out:
                out.write(_LINE.dump_json(line) + b"\n")

    def lines(self) -> list[SpendLine]:
        if not self._path.exists():
            return []
        with self._lock:
            raw = self._path.read_bytes().splitlines()
        try:
            return [_LINE.validate_json(line) for line in raw if line.strip()]
        except ValidationError as err:
            msg = f"{self._path} is damaged; fix it by hand before spending more: {err}"
            raise BudgetError(msg) from err

    def total(self, kind: SpendKind | None = None, service: Service | None = None) -> float:
        return sum(
            line.cost_usd
            for line in self.lines()
            if (kind is None or line.kind == kind) and (service is None or line.service == service)
        )


@dataclass(frozen=True, slots=True)
class Budget:
    cap_usd: float
    reserve_usd: float
    probe_cap_usd: float

    @classmethod
    def from_settings(cls, settings: Settings) -> Budget:
        return cls(
            cap_usd=settings.openrouter_budget_usd,
            reserve_usd=settings.openrouter_reserve_usd,
            probe_cap_usd=settings.probe_budget_usd,
        )

    @property
    def usable_usd(self) -> float:
        return self.cap_usd - self.reserve_usd


def spent_usd(ledger: SpendLedger, key_usage_usd: float | None) -> float:
    """Total spent so far, on every service.

    For OpenRouter, the ledger or the key's own count, whichever is higher; for the rest,
    the ledger.
    """
    openrouter = ledger.total(service=Service.OPENROUTER)
    if key_usage_usd is not None:
        openrouter = max(openrouter, key_usage_usd)
    return openrouter + ledger.total() - ledger.total(service=Service.OPENROUTER)


def check_probe(budget: Budget, ledger: SpendLedger, key_usage_usd: float | None) -> float:
    """Refuse a probe once probes have used their cap. Returns what a probe may still spend."""
    left = budget.probe_cap_usd - ledger.total(SpendKind.PROBE)
    if left <= 0:
        msg = f"probes have spent their ${budget.probe_cap_usd:.2f} cap; raise it to probe again"
        raise BudgetError(msg)
    room = budget.usable_usd - spent_usd(ledger, key_usage_usd)
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
    spent = spent_usd(ledger, key_usage_usd)
    if spent + projected > budget.usable_usd:
        msg = (
            f"{option} on {count} images would cost about ${projected:.2f}. Spent so far: "
            f"${spent:.2f}. Together that passes the ${budget.usable_usd:.2f} usable budget "
            f"(${budget.cap_usd:.2f} cap minus ${budget.reserve_usd:.2f} reserve). "
            "Ask the project lead before raising the openrouter_budget_usd setting."
        )
        raise BudgetError(msg)
    return projected


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
