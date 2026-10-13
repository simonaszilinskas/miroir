"""How close each result card is to the photo or the words: three levels a visitor reads.

Every card shows one level. It comes from what the engine already computes: the first
card's score (for a photo, the routed mix of its raw cosines, as T8 measures "not found";
for words, the fused score) and its gap to the best other card of the top 30. On the tune
halves the gap tells a right first card best; cards after the first are the right record
for 4% of searches or fewer, whatever their score, so they always read "Ressemblance".

- "Très proche": the first card, when its score and gap reach the route's first pair of
  thresholds. Set so that it is the right record at least 80% of the time.
- "Proche": the first card, when it reaches the second, looser pair instead (right about
  two times in five or more).
- "Ressemblance": every other card.

The thresholds are data, one entry per route (`levels.json` in the index bundle, written by
`miroir web levels`). A bundle without the file falls back on the older rule: the first card
is "Très proche" when the search counted as sure, every card else "Ressemblance".
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter

LEVELS_FILE = "levels.json"
#: The route of a search by words alone (photo routes are the routing buckets' names).
WORDS_ROUTE = "words"


class Level(enum.StrEnum):
    VERY_CLOSE = "tres-proche"
    CLOSE = "proche"
    LIKENESS = "ressemblance"


LEVEL_LABELS: dict[Level, str] = {
    Level.VERY_CLOSE: "Très proche",
    Level.CLOSE: "Proche",
    Level.LIKENESS: "Ressemblance",
}


@dataclass(frozen=True, slots=True)
class RouteLevels:
    """The thresholds of one route. A None pair is never reached."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    route: str
    very_close_score: float | None
    """Least score of a "Très proche" first card."""
    very_close_gap: float | None
    """Least gap between that card's score and the best other card's."""
    close_score: float | None
    """Least score of a "Proche" first card."""
    close_gap: float | None
    note: str = ""

    def first_level(self, top: Sequence[float]) -> Level:
        """The first card's level, from the scores of the top cards, best first."""
        if not top:
            return Level.LIKENESS
        gap = gap_of(top)
        if _reaches(top[0], gap, self.very_close_score, self.very_close_gap):
            return Level.VERY_CLOSE
        if _reaches(top[0], gap, self.close_score, self.close_gap):
            return Level.CLOSE
        return Level.LIKENESS

    def card_levels(self, cards: int, top: Sequence[float]) -> list[Level]:
        """The level of each of `cards` cards (best first)."""
        if cards <= 0:
            return []
        return [self.first_level(top), *[Level.LIKENESS] * (cards - 1)]


def _reaches(score: float, gap: float, least_score: float | None, least_gap: float | None) -> bool:
    if least_score is None or least_gap is None:
        return False
    return score >= least_score and gap >= least_gap


def gap_of(scores: Sequence[float]) -> float:
    """First score minus the best of the others; the whole score when alone."""
    if not scores:
        msg = "no scores"
        raise ValueError(msg)
    rest = scores[1:]
    return scores[0] - (max(rest) if rest else 0.0)


@dataclass(frozen=True, slots=True)
class Levels:
    __pydantic_config__ = ConfigDict(extra="forbid")

    routes: tuple[RouteLevels, ...]
    source: str = ""

    def for_route(self, route: str) -> RouteLevels | None:
        return next((r for r in self.routes if r.route == route), None)


_LEVELS = TypeAdapter(Levels)


def read_levels(path: Path) -> Levels | None:
    return _LEVELS.validate_json(path.read_bytes()) if path.exists() else None


def write_levels(path: Path, levels: Levels) -> None:
    path.write_bytes(_LEVELS.dump_json(levels, indent=2) + b"\n")
