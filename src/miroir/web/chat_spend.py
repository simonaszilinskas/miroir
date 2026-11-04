"""What the agentic search (step H5) spends: per conversation, and per UTC day for all."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
