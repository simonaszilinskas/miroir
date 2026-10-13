"""Set the confidence level each result card shows (`miroir.web.levels`), on tune halves.

The site's own ranking on its index, from stored query vectors (NumPy, no model): every tune
photo of T1, T3, T5, T2, T2b (in the index) and T8, T8c (not in it), routed as the server
routes it (type guess, chroma spread), and every T7 tune description for words. Each search
keeps its top `TOP` entities, their scores and which of them is right (`Ranked`, kept on
disk). Then, per route, on a grid of (score, gap) pairs for the first card:

- "Très proche": a pair keeping at least `VERY_CLOSE_TARGET` of the searches it marks
  right (and `SET_FLOOR` on each set with enough of them), and marking at most
  `WRONG_TARGET` of each not-found set (T8, T8c);
- "Proche": a looser pair for the first cards left, keeping at least `CLOSE_TARGET` of them
  right, with at most `CLOSE_WRONG_TARGET` of each not-found set marked "Proche" or
  "Très proche";
- every other card is "Ressemblance".

Of the pairs keeping these promises, the two chosen label the most right first cards
"Très proche" or "Proche" (ties: the more "Très proche" searches).
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter

from miroir.config import WEB_INDEX, Settings, eval_result_path
from miroir.eval.sets import load_queries
from miroir.eval.vectors import read_vectors
from miroir.eval.words import SET_ID as WORDS_SET_ID
from miroir.eval.wordsearch import TEXT_LABEL, WebIndex, text_vectors, word_query_signals
from miroir.web import ranking
from miroir.web.bundle import WRONG_TARGET, SiteBundle, asked_sets, entity_cosines
from miroir.web.levels import (
    LEVELS_FILE,
    WORDS_ROUTE,
    Level,
    Levels,
    RouteLevels,
    gap_of,
    write_levels,
)

logger = logging.getLogger(__name__)

#: Cards kept per search (the gap is measured over them, as the site does).
TOP = 30
VERY_CLOSE_TARGET = 0.80
#: "Très proche" must also be right this often on each set with `MIN_MARKED` of them.
SET_FLOOR = 0.70
CLOSE_TARGET = 0.40
CLOSE_WRONG_TARGET = 0.20
#: Fewest searches a pair must mark to be trusted (and a set must hold to be checked).
MIN_MARKED = 10
#: A not-found set is checked when the route holds this many of its searches.
MIN_NOT_FOUND = 20
PHOTO_SETS = ("T1", "T3", "T5", "T2", "T2b", "T8", "T8c")
NOT_FOUND_SETS = ("T8", "T8c")
SCORE_QUANTILES = tuple(i / 50 for i in range(50))
GAP_GRID = tuple(round(0.005 * i, 3) for i in range(41))  # 0 to 0.2


@dataclass(frozen=True, slots=True)
class Ranked:
    """One tune search as the site ranks it: its route, its top cards' scores (best first)
    and which of them is the right record."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    set_id: str
    query_id: str
    route: str
    scores: tuple[float, ...]
    right: tuple[bool, ...]


_RANKED = TypeAdapter(list[Ranked])


def ranked_path(settings: Settings, index: str = WEB_INDEX) -> Path:
    return settings.matches_path(f"levels-{index}-ranked")


def rank_tune_sets(settings: Settings, index: str = WEB_INDEX) -> list[Ranked]:
    """Every tune search of the photo sets and of T7, ranked as the site does; kept on disk."""
    path = ranked_path(settings, index)
    if path.exists():
        return _RANKED.validate_json(path.read_bytes())
    out = [*_photo_searches(settings, index), *_word_searches(settings)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_RANKED.dump_json(out))
    return out


def _photo_searches(settings: Settings, index: str) -> Iterable[Ranked]:
    bundle = SiteBundle.load(settings.web_index_dir(index))
    catalog, vectors = bundle.catalog, bundle.vectors
    asked = asked_sets(settings, index)
    for set_id in PHOTO_SETS:
        a = asked.get(set_id)
        if a is None or not a.query_ids:
            continue
        e7 = entity_cosines(a.a7, vectors.a7, catalog, a.hidden)
        e3 = vectors.a3_entity_cosines(catalog, a.a7, a.a3, a.hidden)
        for j, query_id in enumerate(a.query_ids):
            photo_type, _ = bundle.guess.photo_type(a.a7[j])
            bucket = bundle.routing.choose(photo_type, a.chroma[j])
            w = bucket.a7_weight
            signals = [
                s for s in (ranking.Signal(e7[j], w), ranking.Signal(e3[j], 1 - w)) if s.weight > 0
            ]
            top = ranking.top_entities(ranking.fuse(signals), TOP)
            right = {catalog.entity_row[e] for e in a.right[j] if e in catalog.entity_row}
            yield Ranked(
                set_id=set_id,
                query_id=query_id,
                route=bucket.name,
                scores=tuple(float(w * e7[j, e] + (1 - w) * e3[j, e]) for e in top),
                right=tuple(int(e) in right for e in top),
            )
        logger.info("%s: %d searches ranked", set_id, len(a.query_ids))


def _word_searches(settings: Settings) -> Iterable[Ranked]:
    index = WebIndex.load(settings)
    weights = SiteBundle.load(index.folder).routing.text
    run = read_vectors(settings.vectors_path(f"A7-embeddinggemma-2-512-{TEXT_LABEL}"))
    queries = load_queries(settings, WORDS_SET_ID, "tune")
    signals = word_query_signals(index, queries, text_vectors(run, queries))
    for j, query_id in enumerate(signals.query_ids):
        fused = ranking.fuse(
            [
                s
                for s in (
                    ranking.Signal(signals.a7[j], weights.a7),
                    ranking.Signal(signals.bm25[j], weights.bm25),
                )
                if s.weight > 0
            ]
        )
        top = ranking.top_entities(fused, TOP)
        yield Ranked(
            set_id=WORDS_SET_ID,
            query_id=query_id,
            route=WORDS_ROUTE,
            scores=tuple(float(fused[e]) for e in top),
            right=tuple(int(e) == int(signals.right[j]) for e in top),
        )
    logger.info("%s: %d searches ranked", WORDS_SET_ID, len(signals.query_ids))


# --- choosing the thresholds -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Firsts:
    """The first card of each search of one route: score, gap, right or not, set."""

    score: NDArray[np.float64]
    gap: NDArray[np.float64]
    right: NDArray[np.bool_]
    set_id: NDArray[np.str_]

    @classmethod
    def of(cls, rows: Sequence[Ranked]) -> Firsts:
        return cls(
            score=np.asarray([r.scores[0] for r in rows], dtype=np.float64),
            gap=np.asarray([gap_of(r.scores) for r in rows], dtype=np.float64),
            right=np.asarray([r.right[0] for r in rows], dtype=np.bool_),
            set_id=np.asarray([r.set_id for r in rows], dtype=np.str_),
        )

    def marked(self, score: float | None, gap: float | None) -> NDArray[np.bool_]:
        if score is None or gap is None:
            return np.zeros(len(self.score), dtype=np.bool_)
        return (self.score >= score) & (self.gap >= gap)

    @property
    def lost(self) -> NDArray[np.bool_]:
        return np.isin(self.set_id, NOT_FOUND_SETS)


def _pairs(firsts: Firsts) -> list[tuple[float, float]]:
    asked = firsts.score[~firsts.lost]
    scores = sorted({round(float(v), 4) for v in np.quantile(asked, SCORE_QUANTILES)})
    return [(s, g) for g in GAP_GRID for s in scores]


def _honest(firsts: Firsts, marked: NDArray[np.bool_], target: float) -> bool:
    """At most `target` of each not-found set the route holds enough of is marked."""
    for set_id in NOT_FOUND_SETS:
        these = firsts.set_id == set_id
        if these.sum() >= MIN_NOT_FOUND and marked[these].mean() > target:
            return False
    return True


def choose_route(route: str, rows: Sequence[Ranked]) -> RouteLevels:
    """The two pairs labelling the most right first cards "Très proche" or "Proche", each
    keeping its promises (module doc); ties go to the more "Très proche" searches."""
    firsts = Firsts.of(rows)
    asked = ~firsts.lost
    pairs = _pairs(firsts)
    marks = np.stack([firsts.marked(score, gap) for score, gap in pairs])
    # "Très proche" candidates, one per distinct set of marked searches.
    candidates: dict[bytes, int] = {}
    for i in range(len(pairs)):
        mine = marks[i] & asked
        if mine.sum() < MIN_MARKED or firsts.right[mine].mean() < VERY_CLOSE_TARGET:
            continue
        if _honest(firsts, marks[i], WRONG_TARGET) and _each_set(firsts, mine):
            candidates.setdefault(mine.tobytes(), i)
    best: tuple[int, int, int | None, int | None] = (-1, -1, None, None)
    for very_close in [*candidates.values(), None]:
        taken = np.zeros(len(asked), dtype=np.bool_) if very_close is None else marks[very_close]
        right = int((taken & asked & firsts.right).sum())
        close = _close(firsts, marks, taken)
        if close is not None:
            right += int((marks[close] & ~taken & asked & firsts.right).sum())
        key = (right, int((taken & asked).sum()), very_close, close)
        if key[:2] > best[:2]:
            best = key
    _, _, very_close, close = best
    return RouteLevels(
        route,
        very_close_score=None if very_close is None else pairs[very_close][0],
        very_close_gap=None if very_close is None else pairs[very_close][1],
        close_score=None if close is None else pairs[close][0],
        close_gap=None if close is None else pairs[close][1],
    )


def _close(firsts: Firsts, marks: NDArray[np.bool_], taken: NDArray[np.bool_]) -> int | None:
    """The "Proche" pair marking the most right first cards that "Très proche" left."""
    left = ~firsts.lost & ~taken
    mine = marks & left
    count = mine.sum(axis=1)
    right = (mine & firsts.right).sum(axis=1)
    share = np.divide(right, count, out=np.zeros(len(count)), where=count > 0)
    ok = (count >= MIN_MARKED) & (share >= CLOSE_TARGET)
    for set_id in NOT_FOUND_SETS:
        these = firsts.set_id == set_id
        if these.sum() >= MIN_NOT_FOUND:
            ok &= ((marks | taken)[:, these]).mean(axis=1) <= CLOSE_WRONG_TARGET
    if not ok.any():
        return None
    return int(np.flatnonzero(ok)[np.argmax(right[ok])])


def _each_set(firsts: Firsts, marked: NDArray[np.bool_]) -> bool:
    for set_id in np.unique(firsts.set_id[marked]):
        these = marked & (firsts.set_id == set_id)
        if these.sum() >= MIN_MARKED and firsts.right[these].mean() < SET_FLOOR:
            return False
    return True


@dataclass(frozen=True, slots=True)
class LevelCount:
    """How the first card fares at one level: share of searches, share of those right."""

    searches: float
    right: float


def level_count(rows: Sequence[Ranked], rule: RouteLevels, level: Level) -> LevelCount:
    hits = [r.right[0] for r in rows if rule.first_level(r.scores) is level]
    return LevelCount(
        searches=len(hits) / len(rows) if rows else 0.0,
        right=sum(hits) / len(hits) if hits else 0.0,
    )


@dataclass(frozen=True, slots=True)
class RouteReport:
    route: str
    rule: RouteLevels
    searches: int
    """In-index searches of the route (right record in the index)."""
    levels: dict[str, LevelCount]
    """Per level of the first card, over the route's in-index searches."""
    per_set: dict[str, dict[str, LevelCount]]
    """The same per set; for not-found sets, `right` is 0 and `searches` is what matters."""


def report_route(route: str, rows: Sequence[Ranked], rule: RouteLevels) -> RouteReport:
    asked = [r for r in rows if r.set_id not in NOT_FOUND_SETS]
    per_set = {
        set_id: {lv.value: level_count(these, rule, lv) for lv in Level}
        for set_id in sorted({r.set_id for r in rows})
        if (these := [r for r in rows if r.set_id == set_id])
    }
    return RouteReport(
        route=route,
        rule=rule,
        searches=len(asked),
        levels={lv.value: level_count(asked, rule, lv) for lv in Level},
        per_set=per_set,
    )


def choose_levels(rows: Sequence[Ranked]) -> list[RouteReport]:
    reports: list[RouteReport] = []
    for route in dict.fromkeys(r.route for r in rows):
        these = [r for r in rows if r.route == route]
        reports.append(report_route(route, these, choose_route(route, these)))
    return reports


def not_found_shares(rows: Sequence[Ranked], reports: Sequence[RouteReport]) -> dict[str, float]:
    """Per not-found set, over every route: the share of searches whose first card shows
    "Très proche" (each search read with its own route's rule)."""
    rules = {r.route: r.rule for r in reports}
    out: dict[str, float] = {}
    for set_id in NOT_FOUND_SETS:
        lost = [r for r in rows if r.set_id == set_id and r.route in rules]
        if lost:
            shown = [rules[r.route].first_level(r.scores) is Level.VERY_CLOSE for r in lost]
            out[set_id] = sum(shown) / len(lost)
    return out


def write_level_rules(
    settings: Settings, rows: Sequence[Ranked], *, round_id: str = "R3"
) -> list[RouteReport]:
    """Write `levels.json` into the bundle, keep a copy for `build-index`, record the numbers."""
    reports = choose_levels(rows)
    date = datetime.now(UTC).date().isoformat()
    levels = Levels(
        routes=tuple(replace(r.rule, note=_note(r, date)) for r in reports),
        source=f"miroir web levels, index {WEB_INDEX}, {date}",
    )
    bundle = settings.web_index_dir(WEB_INDEX)
    write_levels(bundle / LEVELS_FILE, levels)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / LEVELS_FILE, settings.index_dir / f"levels-{WEB_INDEX}.json")
    detail = eval_result_path(round_id, "G", "levels").parent / "G-detail" / "levels.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(
        json.dumps(
            {
                "index": WEB_INDEX,
                "date": date,
                "not_found_very_close": not_found_shares(rows, reports),
                "routes": [_report(r) for r in reports],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return reports


def _note(r: RouteReport, date: str) -> str:
    vc, cl = r.levels[Level.VERY_CLOSE.value], r.levels[Level.CLOSE.value]
    text = (
        f"Tune halves, index {WEB_INDEX}, {date}, {r.searches} searches: 'Très proche' on "
        f"{vc.searches:.0%}, right {vc.right:.0%}; 'Proche' on {cl.searches:.0%}, right "
        f"{cl.right:.0%}"
    )
    for set_id in NOT_FOUND_SETS:
        if set_id in r.per_set:
            text += (
                f"; {set_id} 'Très proche' {r.per_set[set_id][Level.VERY_CLOSE.value].searches:.0%}"
            )
    return text


def _report(r: RouteReport) -> dict[str, object]:
    def counts(levels: dict[str, LevelCount]) -> dict[str, dict[str, float]]:
        return {k: {"searches": v.searches, "right": v.right} for k, v in levels.items()}

    return {
        "route": r.route,
        "very_close_score": r.rule.very_close_score,
        "very_close_gap": r.rule.very_close_gap,
        "close_score": r.rule.close_score,
        "close_gap": r.rule.close_gap,
        "searches": r.searches,
        "levels": counts(r.levels),
        "per_set": {k: counts(v) for k, v in r.per_set.items()},
    }
