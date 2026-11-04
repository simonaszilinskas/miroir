"""What the agentic search (step H5) spends: per conversation, and per UTC day for all.

A call is checked against the caps before it is made, and its cost added once it is known.
In between, `Budget` holds its guessed cost, so calls made at once cannot each pass a cap
that had room for one. The holds live in the site's process, where every conversation runs;
a `SpendStore` only adds real costs (`LedgerSpend` writes each one to the project ledger).
A site of several processes sharing one store would need the holds in the store too.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class SpendStore(Protocol):
    """Dollars spent per UTC day, shared by every conversation."""

    async def spent(self, day: str) -> float: ...
    async def add(self, day: str, usd: float) -> float:
        """Adds and returns the day's new total."""
        ...


class MemorySpend:
    def __init__(self) -> None:
        self._days: dict[str, float] = {}

    async def spent(self, day: str) -> float:
        return self._days.get(day, 0.0)

    async def add(self, day: str, usd: float) -> float:
        self._days[day] = self._days.get(day, 0.0) + usd
        return self._days[day]


class FileSpend:
    """Per-day totals in a small JSON file, so a restarted site keeps the day's."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _days(self) -> dict[str, float]:
        if not self._path.exists():
            return {}
        raw: object = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {str(k): float(str(v)) for k, v in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

    async def spent(self, day: str) -> float:
        return self._days().get(day, 0.0)

    async def add(self, day: str, usd: float) -> float:
        days = self._days()
        days[day] = days.get(day, 0.0) + usd
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(days, indent=1) + "\n", encoding="utf-8")
        return days[day]


@dataclass(frozen=True, slots=True)
class Limits:
    conversation_usd: float
    day_usd: float


class Cap(StrEnum):
    """The cap a call would pass."""

    CONVERSATION = "conversation"
    DAY = "day"


@dataclass(frozen=True, slots=True)
class Hold:
    """A call's guessed cost, held against the day's cap until its real cost is known."""

    day: str
    usd: float


class Budget:
    """The caps of `Limits`, with the calls in flight counted."""

    def __init__(self, store: SpendStore, limits: Limits, today: Callable[[], str]) -> None:
        self.store = store
        self.limits = limits
        self._today = today
        self._held: dict[str, float] = {}

    def held(self, day: str) -> float:
        return self._held.get(day, 0.0)

    async def hold(self, conversation_usd: float, guess: float) -> Hold | Cap:
        """Hold `guess` for a call, or the cap it could pass (conversation, then day)."""
        if conversation_usd + guess > self.limits.conversation_usd:
            return Cap.CONVERSATION
        day = self._today()
        spent = await self.store.spent(day)
        # No await from here to the hold: another call's check sees it.
        if spent + self.held(day) + guess > self.limits.day_usd:
            return Cap.DAY
        self._held[day] = self.held(day) + guess
        return Hold(day, guess)

    async def settle(self, hold: Hold, usd: float) -> float:
        """The call's real cost replaces its hold. Returns the day's new total."""
        try:
            return await self.store.add(hold.day, usd)
        finally:
            self.release(hold)

    def release(self, hold: Hold) -> None:
        """A held call that cost nothing (refused before it ran)."""
        left = self.held(hold.day) - hold.usd
        if left > 1e-12:
            self._held[hold.day] = left
        else:
            self._held.pop(hold.day, None)
