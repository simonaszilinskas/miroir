"""G5: G4 with three buckets: archive building, colour building, object (experiments.md G).

G4 gives every building photo one EmbeddingGemma 2 (A7) weight, tuned on T1's archive
prints. A visitor's colour photo of a building (T2) wants MegaLoc (A3) instead. G5 splits
buildings in two with a pixel statistic of the query photo (`images.colour.chroma_spread`,
no model): black-and-white or tinted is an archive building, else a colour building. Each
of the three buckets gets its own first-model weight, as a hard switch.

The weights come from the G4 grid, one per bucket, picked on the tune halves of every scored
set together (mean hit@10 over sets, then mAP; a tie goes to the weight nearest 0.5). A
query's ranking depends only on its own weight, so each set is ranked once per grid weight
(`SetGrid`) and every policy (G1, G4, G5, a model alone) is read from those rankings. The
split-half check of G4 (`router.cross_fitted`) is done the same way: weights picked on half
the buildings and objects, scored on the other half.

Queries that are Commons photos (T2) are asked with each model's query run; a query run also
overrides gallery vectors for ids it holds (F1: grey copies of T1 queries).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from enum import StrEnum

import httpx
import numpy as np
from numpy.typing import NDArray

from miroir.commons.images import open_commons_images
from miroir.commons.model import is_commons_image
from miroir.config import SEED, Settings, eval_result_path
from miroir.eval.fusion import min_max
from miroir.eval.results import RunResult, SetScore, set_score, write_result
from miroir.eval.router import (
    QUERY_SETS,
    ROUTE_GRID,
    SWITCH_AT,
    GuessReport,
    Route,
    RoutedSet,
    features,
    guess_report,
    photo_types,
    rank_routed,
)
from miroir.eval.run import entity_map, publishable_only, search_queries
from miroir.eval.scoring import (
    Difference,
    ScoreResult,
    entity_scores,
    paired_bootstrap,
    per_query_values,
)
from miroir.eval.sets import load_queries, read_gallery
from miroir.eval.stack import Model, combined_info, load_model
from miroir.eval.vectors import VectorSet, read_vectors
from miroir.images.colour import GREY_BELOW, chroma_spread
from miroir.images.fetch import ImageCache, ImageFetcher
from miroir.records.store import read_images

logger = logging.getLogger(__name__)

Float64s = NDArray[np.float64]


class Bucket(StrEnum):
    ARCHIVE_BUILDING = "archive-building"
    COLOUR_BUILDING = "colour-building"
    OBJECT = "object"


def bucket_of(p_building: float, chroma: float) -> Bucket:
    if p_building <= SWITCH_AT:
        return Bucket.OBJECT
    return Bucket.ARCHIVE_BUILDING if chroma < GREY_BELOW else Bucket.COLOUR_BUILDING


#: How a policy groups buckets: queries in one group share a weight.
type Grouping = Callable[[Bucket], str]


def three(b: Bucket) -> str:
    """G5: one weight per bucket."""
    return b.value


def two(b: Bucket) -> str:
    """G4: one weight for buildings, one for objects."""
    return "object" if b is Bucket.OBJECT else "building"


def one(_: Bucket) -> str:
    """G1: one weight for every photo."""
    return "all"


@dataclass(frozen=True, slots=True)
class SetGrid:
    """One set's tune queries ranked at every grid weight, with each query's bucket."""

    set_id: str
    results: tuple[ScoreResult, ...]
    """One per `ROUTE_GRID` weight; the same scored queries, in the same order, in each."""
    buckets: tuple[Bucket, ...]
    entities: tuple[str, ...]
    """The building or object each scored query shows (folds split on it)."""
    query_images: tuple[str, ...]

    def __post_init__(self) -> None:
        ids = [tuple(q.query_id for q in r.per_query) for r in self.results]
        if len(set(ids)) != 1 or len(ids[0]) != len(self.buckets):
            msg = f"{self.set_id}: grid rankings score different queries"
            raise ValueError(msg)

    @property
    def size(self) -> int:
        return len(self.buckets)

    def table(self, metric: str) -> Float64s:
        """(grid weights, queries) array of one metric."""
        return np.stack([per_query_values(r.per_query)[metric] for r in self.results])

    def at(self, weight_of: Mapping[Bucket, float]) -> ScoreResult:
        """The set scored with each query at its bucket's weight."""
        column = {w: i for i, w in enumerate(ROUTE_GRID)}
        per_query = tuple(
            self.results[column[weight_of[b]]].per_query[j] for j, b in enumerate(self.buckets)
        )
        return replace(self.results[0], per_query=per_query)


def pick_weights(
    grids: Sequence[SetGrid],
    grouping: Grouping,
    masks: Sequence[NDArray[np.bool_]] | None = None,
) -> dict[Bucket, float]:
    """Each group's grid weight with the best mean hit@10 over sets (then mAP; then nearest 0.5).

    A query's ranking depends only on its own weight, so the mean over sets splits into one
    term per group and each group is picked alone. `masks` keeps a subset of each set's
    queries (the split-half check).
    """
    kept = masks or [np.ones(g.size, dtype=bool) for g in grids]
    totals: dict[str, NDArray[np.float64]] = {}
    for grid, keep in zip(grids, kept, strict=True):
        n = int(keep.sum())
        if n == 0:
            continue
        hit, ap = grid.table("hit@10"), grid.table("map")
        for group in {grouping(b) for b in grid.buckets}:
            rows = keep & np.asarray([grouping(b) == group for b in grid.buckets], dtype=np.bool_)
            # One row per weight: [hit@10 share, mAP share] that this group adds to the mean.
            add = np.column_stack([hit[:, rows].sum(axis=1), ap[:, rows].sum(axis=1)]) / n
            totals[group] = totals.get(group, np.zeros((len(ROUTE_GRID), 2))) + add
    chosen: dict[str, float] = {}
    for group, score in totals.items():
        order = sorted(range(len(ROUTE_GRID)), key=lambda i: abs(ROUTE_GRID[i] - 0.5))
        best = max(order, key=lambda i: (round(score[i, 0], 12), round(score[i, 1], 12)))
        chosen[group] = ROUTE_GRID[best]
    default = 0.5
    return {b: chosen.get(grouping(b), default) for b in Bucket}


def split_half(grids: Sequence[SetGrid], grouping: Grouping, seed: int = SEED) -> dict[str, float]:
    """Hit@10 per set when each half's weights were picked on the other half (by entity)."""
    entities = sorted({e for g in grids for e in g.entities})
    random.Random(seed).shuffle(entities)
    fold_of = {e: i % 2 for i, e in enumerate(entities)}
    folds = [np.asarray([fold_of[e] for e in g.entities]) for g in grids]
    hits: dict[str, list[float]] = {g.set_id: [] for g in grids}
    for fold in (0, 1):
        weights = pick_weights(grids, grouping, [f != fold for f in folds])
        for grid, f in zip(grids, folds, strict=True):
            scored = per_query_values(grid.at(weights).per_query)["hit@10"]
            hits[grid.set_id].extend(scored[f == fold].tolist())
    return {s: float(np.mean(v)) for s, v in hits.items()}


# --- building the grids ----------------------------------------------------------------------


def asking_run(
    gallery_run: VectorSet, query_run: VectorSet | None, ids: Sequence[str]
) -> VectorSet:
    """A small run holding the vectors of `ids`: from the query run when it has them, else
    from the gallery run."""
    sources: list[tuple[VectorSet, dict[str, int]]] = []
    if query_run is not None:
        sources.append((query_run, query_run.index()))
    sources.append((gallery_run, gallery_run.index()))
    rows: list[NDArray[np.float16]] = []
    kept: list[str] = []
    for image_id in dict.fromkeys(ids):
        for run, row_of in sources:
            if image_id in row_of:
                rows.append(np.asarray(run.vectors[row_of[image_id]], dtype=np.float16))
                kept.append(image_id)
                break
    missing = len(set(ids)) - len(kept)
    if missing:
        msg = f"{gallery_run.info.option}: {missing} query image(s) have no vector"
        raise ValueError(msg)
    return VectorSet(ids=tuple(kept), vectors=np.stack(rows), info=gallery_run.info)


def query_chroma(settings: Settings, image_ids: Sequence[str]) -> dict[str, float]:
    """`chroma_spread` of each query photo: POP images from the image cache (fetched when
    missing), Commons photos from the Commons cache. Fails if one cannot be read."""
    out: dict[str, float] = {}
    pop_ids = [i for i in image_ids if not is_commons_image(i)]
    commons_ids = [i for i in image_ids if is_commons_image(i)]
    images = {i.image_id: i for i in read_images(settings.images_parquet)}
    cache = ImageCache(settings.image_cache_dir, settings.image_cache_mb * 1_000_000)
    with httpx.Client(timeout=60.0) as http:
        fetcher = ImageFetcher(http, cache)
        for fetched in fetcher.fetch_all(images[i] for i in pop_ids):
            if fetched.jpeg is not None:
                out[fetched.image_id] = chroma_spread(fetched.jpeg)
    with open_commons_images(settings) as commons:
        for fetched in commons.fetch_all(commons_ids):
            if fetched.jpeg is not None:
                out[fetched.image_id] = chroma_spread(fetched.jpeg)
    missing = [i for i in image_ids if i not in out]
    if missing:
        msg = f"{len(missing)} query photo(s) could not be read, e.g. {missing[:3]}"
        raise RuntimeError(msg)
    return out


@dataclass(frozen=True, slots=True)
class Grids:
    sets: tuple[SetGrid, ...]
    models: tuple[Model, ...]
    guess: GuessReport
    query_runs: tuple[str, ...]


def build_grids(
    settings: Settings,
    *,
    runs: Sequence[str],
    query_runs: Sequence[str],
    set_ids: Sequence[str],
    gallery: str,
    publishable_queries: bool,
) -> Grids:
    """Rank each set's tune queries at every grid weight. The guess reads the first model."""
    if len(runs) != 2 or (query_runs and len(query_runs) != len(runs)):
        msg = "G5 takes two gallery runs, and one query run per model or none"
        raise ValueError(msg)
    gallery_ids = read_gallery(settings, gallery)
    entity_of = entity_map(settings)
    models = tuple(load_model(settings, r, entity_of, gallery_ids) for r in runs)
    asked_runs = [read_vectors(settings.vectors_path(r)) for r in query_runs] or [None, None]
    guess = guess_report(
        settings,
        (models[0].vectors,),
        gallery_ids,
        entity_of,
        photo_types(settings),
        query_sets=(*QUERY_SETS, *(s for s in set_ids if s not in QUERY_SETS)),
    )
    loaded = {s: load_queries(settings, s, "tune") for s in set_ids}
    if publishable_queries:
        loaded = {
            s: publishable_only(settings, q, in_gallery=gallery_ids) for s, q in loaded.items()
        }
    all_ids = [q.query_image_id for qs in loaded.values() for q in qs]
    asking = [asking_run(m.vectors, a, all_ids) for m, a in zip(models, asked_runs, strict=True)]
    chroma = query_chroma(settings, all_ids)
    p_of = dict(
        zip(
            asking[0].ids,
            guess.guess.p_building(features((asking[0],), asking[0].ids)).tolist(),
            strict=True,
        )
    )
    grids: list[SetGrid] = []
    for set_id, qs in loaded.items():
        queries = tuple(search_queries(qs))
        parts = tuple(
            entity_scores(a, m.gallery, queries) for a, m in zip(asking, models, strict=True)
        )
        p = np.asarray([p_of[q.image_id] for q in queries])
        routed = RoutedSet(queries, parts, tuple(min_max(x.values) for x in parts), p)
        results = tuple(rank_routed(routed, Route(w, w)) for w in ROUTE_GRID)
        scored = {q.query_id for q in results[0].per_query}
        by_id = {q.query_id: q for q in qs}
        kept = [q for q in queries if q.query_id in scored]
        grids.append(
            SetGrid(
                set_id=set_id,
                results=results,
                buckets=tuple(bucket_of(p_of[q.image_id], chroma[q.image_id]) for q in kept),
                entities=tuple(by_id[q.query_id].entity_id for q in kept),
                query_images=tuple(q.image_id for q in kept),
            )
        )
        counts = {b.value: grids[-1].buckets.count(b) for b in Bucket}
        logger.info(
            "%s: %d queries ranked at %d weights; buckets %s",
            set_id,
            len(kept),
            len(ROUTE_GRID),
            counts,
        )
    return Grids(tuple(grids), models, guess, tuple(query_runs))


# --- one run ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Policy:
    """A named way to weight A7 per query, and the weights it used."""

    name: str
    weights: Mapping[Bucket, float]
    split_half: Mapping[str, float] | None = None


def policies(grids: Sequence[SetGrid], fixed: Mapping[Bucket, float] | None) -> list[Policy]:
    """Each model alone, G1, G4 and G5 picked on these sets, and `fixed` if given."""
    out = [
        Policy("A7 alone", dict.fromkeys(Bucket, 1.0)),
        Policy("A3 alone", dict.fromkeys(Bucket, 0.0)),
    ]
    for name, grouping in (("G1", one), ("G4", two), ("G5", three)):
        out.append(Policy(name, pick_weights(grids, grouping), split_half(grids, grouping)))
    if fixed is not None:
        out.append(Policy("fixed", fixed))
    return out


@dataclass(frozen=True, slots=True)
class Router3Report:
    result: RunResult
    policies: tuple[Policy, ...]
    scores: Mapping[str, Mapping[str, SetScore]]
    """Policy name, then set: its score."""
    differences: Mapping[str, Mapping[str, Difference]]
    """Key "<chosen> - <policy>", then set: paired hit@10 difference."""
    buckets: Mapping[str, Mapping[str, int]]


def score_router3(
    settings: Settings,
    *,
    runs: Sequence[str],
    query_runs: Sequence[str],
    set_ids: Sequence[str],
    gallery: str,
    round_id: str,
    step: str,
    option: str,
    publishable_queries: bool = True,
    fixed: Mapping[Bucket, float] | None = None,
) -> Router3Report:
    """Rank the sets, pick G1, G4 and G5 weights on them, write the chosen policy's scores.

    The chosen policy is `fixed` when given (weights picked elsewhere), else G5.
    """
    grids = build_grids(
        settings,
        runs=runs,
        query_runs=query_runs,
        set_ids=set_ids,
        gallery=gallery,
        publishable_queries=publishable_queries,
    )
    found = policies(grids.sets, fixed)
    scores = {
        p.name: {
            g.set_id: set_score(g.at(p.weights), test_set=g.set_id, half="tune", gallery=gallery)
            for g in grids.sets
        }
        for p in found
    }
    chosen = found[-1]
    differences = {
        f"{chosen.name} - {p.name}": {
            s: _hit10(
                paired_bootstrap(scores[chosen.name][s].per_query, scores[p.name][s].per_query)
            )
            for s in set_ids
        }
        for p in found
        if p is not chosen
    }
    notes = _notes(grids, found, chosen)
    run_name = "+".join(runs)
    result = write_result(
        eval_result_path(round_id, step, option),
        round_id=round_id,
        step=step,
        option=option,
        vector_run=run_name,
        query_vector_run="+".join(query_runs) if query_runs else run_name,
        vectors=combined_info([m.vectors.info for m in grids.models], notes),
        scores=list(scores[chosen.name].values()),
        today=_today(),
    )
    buckets = {g.set_id: {b.value: g.buckets.count(b) for b in Bucket} for g in grids.sets}
    return Router3Report(result, tuple(found), scores, differences, buckets)


def _hit10(differences: Sequence[Difference]) -> Difference:
    return next(d for d in differences if d.metric == "hit@10")


def _weights_text(weights: Mapping[Bucket, float]) -> str:
    return ", ".join(f"{b.value} {weights[b]:.1f}" for b in Bucket)


def _notes(grids: Grids, found: Sequence[Policy], chosen: Policy) -> str:
    first = grids.models[0].vectors.info.option
    second = grids.models[1].vectors.info.option
    report = grids.guess
    accuracy = ", ".join(f"{s} {100 * a:.1f}% of {n}" for s, n, a in report.query_accuracy)
    parts = [
        f"{chosen.name}: {first} weight per bucket ({_weights_text(chosen.weights)}), "
        f"{second} the rest, hard switch; weighted sum of min-max scores. Buckets: "
        f"p(building) > {SWITCH_AT} from a logistic regression on {first} vectors "
        f"(penalty {report.guess.penalty:g}, {report.trained_on} gallery images, "
        f"{report.left_out_entities} query entities left out, right on tune queries: "
        f"{accuracy}); a building photo is archive when its chroma spread (std of Cb, Cr) "
        f"is under {GREY_BELOW}, else colour. Weights picked from the grid by mean hit@10 over "
        f"{', '.join(g.set_id for g in grids.sets)} (tune halves)",
    ]
    for p in found:
        if p.split_half is None:
            continue
        halves = ", ".join(f"{s} {100 * h:.1f}" for s, h in p.split_half.items())
        parts.append(f"{p.name} ({_weights_text(p.weights)}) split-half hit@10: {halves}")
    return "; ".join(parts)


def _today() -> date:
    return datetime.now(UTC).date()
