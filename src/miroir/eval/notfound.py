"""The "not found" measure on T8 (experiments.md step G, technical spec §6.3 and §7).

Each query's confidence is the score of its top entity. A cut-off on it decides "match" or
"not found". T8 queries have their whole entity taken out of the gallery, so any match on
them is wrong; in-index queries (T1 and T5, whose entities are in the gallery) should match.

Two numbers, from one set of confidences:

- wrong-but-confident rate at a fixed match rate: put the cut-off where `MATCH_RATE` of
  in-index queries match, and count the share of T8 queries that match too (experiments.md G);
- match rate at a fixed wrong-but-confident rate: the lowest cut-off that keeps T8 matches
  under `WRONG_TARGET`, and the share of in-index queries it lets through (technical spec §6.3).

Both get a 95% bootstrap interval, resampling in-index and T8 queries apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

from miroir.config import SEED
from miroir.eval.scoring import RESAMPLES, EntityScores, Interval

#: Share of in-index queries that match at the cut-off of the first measure.
MATCH_RATE = 0.5
#: Highest share of T8 queries allowed to match, for the second measure (technical spec §7).
WRONG_TARGET = 0.05

_EPS = 1e-9

Values = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Confidence:
    query_id: str
    score: float


@dataclass(frozen=True, slots=True)
class NotFoundScore:
    """One option's "not found" measure on one T8 half, against one gallery."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    test_set: str
    """The set without answers, "T8"."""
    half: str
    gallery: str
    in_index_sets: tuple[str, ...]
    """Sets whose queries stand for searches that should match, e.g. ("T1", "T5")."""
    match_rate: float
    wrong_confident: Interval
    """Share of T8 queries at or above the cut-off that lets `match_rate` of in-index match."""
    wrong_target: float
    match_rate_at_target: Interval
    """Share of in-index queries above the lowest cut-off that keeps T8 under `wrong_target`."""
    in_index: tuple[Confidence, ...]
    not_found: tuple[Confidence, ...]

    def __post_init__(self) -> None:
        if not self.in_index or not self.not_found:
            msg = f"{self.test_set}: needs in-index and not-found queries"
            raise ValueError(msg)
        found, missing = _values(self.in_index), _values(self.not_found)
        checks = (
            (self.wrong_confident, wrong_confident_rate(found, missing, self.match_rate)),
            (self.match_rate_at_target, match_rate_at(found, missing, self.wrong_target)),
        )
        for interval, value in checks:
            if abs(interval.value - value) > _EPS:
                msg = f"{self.test_set}: stored {interval.value} != {value} from the confidences"
                raise ValueError(msg)
            if not 0 <= interval.low <= interval.high <= 1:
                msg = f"{self.test_set}: interval {interval} is out of order"
                raise ValueError(msg)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.test_set, self.half, self.gallery)


def confidences(
    ranking: EntityScores,
    raw: Sequence[EntityScores],
    weights: Sequence[float] | NDArray[np.floating],
) -> tuple[Confidence, ...]:
    """Each query's confidence: the weighted raw scores of the entity ranked first.

    For one model, pass its scores as `ranking` and `raw` with weight 1: the top cosine.
    After fusion, the fused scores pick the top entity but say little about confidence (G1
    gives the top entity about 1 whatever happens), so the models' own cosines score it.
    `weights` is one weight per model, or one row per query when weights vary by query (G4).
    """
    top = np.argmax(ranking.values, axis=1)
    rows = np.arange(len(ranking.query_ids))
    per_query = np.broadcast_to(np.asarray(weights, dtype=np.float64), (len(rows), len(raw)))
    total = np.zeros(len(rows), dtype=np.float64)
    for m, part in enumerate(raw):
        if part.query_ids != ranking.query_ids or part.entity_ids != ranking.entity_ids:
            msg = "raw scores and ranking cover different queries or entities"
            raise ValueError(msg)
        total += per_query[:, m] * part.values[rows, top].astype(np.float64)
    return tuple(Confidence(q, float(s)) for q, s in zip(ranking.query_ids, total, strict=True))


def wrong_confident_rate(found: Values, missing: Values, match_rate: float = MATCH_RATE) -> float:
    if not 0 < match_rate <= 1:
        msg = f"match rate must be in (0, 1], got {match_rate}"
        raise ValueError(msg)
    cutoff = np.sort(found)[::-1][math.ceil(match_rate * found.size) - 1]
    return float((missing >= cutoff).mean())


def match_rate_at(found: Values, missing: Values, wrong_target: float = WRONG_TARGET) -> float:
    allowed = math.floor(wrong_target * missing.size)
    if allowed >= missing.size:
        return 1.0
    bar = np.sort(missing)[::-1][allowed]
    return float((found > bar).mean())


def not_found_score(
    in_index: Sequence[Confidence],
    not_found: Sequence[Confidence],
    *,
    test_set: str,
    half: str,
    gallery: str,
    in_index_sets: Sequence[str],
    seed: int = SEED,
) -> NotFoundScore:
    found, missing = _values(in_index), _values(not_found)
    rng = np.random.default_rng(seed)
    wrong: list[float] = []
    matched: list[float] = []
    for _ in range(RESAMPLES):
        f = found[rng.integers(0, found.size, found.size)]
        m = missing[rng.integers(0, missing.size, missing.size)]
        wrong.append(wrong_confident_rate(f, m))
        matched.append(match_rate_at(f, m))
    return NotFoundScore(
        test_set=test_set,
        half=half,
        gallery=gallery,
        in_index_sets=tuple(in_index_sets),
        match_rate=MATCH_RATE,
        wrong_confident=_interval(wrong_confident_rate(found, missing), wrong),
        wrong_target=WRONG_TARGET,
        match_rate_at_target=_interval(match_rate_at(found, missing), matched),
        in_index=tuple(in_index),
        not_found=tuple(not_found),
    )


def _interval(value: float, resampled: list[float]) -> Interval:
    low, high = np.percentile(np.asarray(resampled), [2.5, 97.5])
    return Interval(value=value, low=float(low), high=float(high))


def _values(scores: Sequence[Confidence]) -> Values:
    return np.asarray([c.score for c in scores], dtype=np.float64)
