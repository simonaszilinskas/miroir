"""G5's buckets and grids (experiments.md G): archive building, colour building, object.

A query photo's bucket comes from the building-or-object guess (`miroir.eval.router`) and,
for a building, a pixel statistic of the photo (`images.colour.chroma_spread`, no model):
black-and-white or tinted is an archive building, else a colour building. Each bucket has its
own EmbeddingGemma 2 (A7) weight, MegaLoc (A3) the rest (the site's weights,
`miroir.web.routing`).

`build_grids` ranks each set's tune queries once per `ROUTE_GRID` weight (`SetGrid`), so any
weight per bucket is read from those rankings (`SetGrid.at`); step H4 scores G5 that way.
Queries that are Commons photos (T2) are asked with each model's query run (`asking_run`).
The G5 weight search itself is retired (docs/experiments.md, "Retired code").
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import httpx
import numpy as np
from numpy.typing import NDArray

from miroir.commons.images import open_commons_images
from miroir.commons.model import is_commons_image
from miroir.config import Settings
from miroir.eval.fusion import min_max
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
from miroir.eval.run import Model, entity_map, load_model, publishable_only, search_queries
from miroir.eval.scoring import (
    ScoreResult,
    entity_scores,
)
from miroir.eval.sets import load_queries, read_gallery
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


@dataclass(frozen=True, slots=True)
class SetGrid:
    """One set's tune queries ranked at every grid weight, with each query's bucket."""

    set_id: str
    results: tuple[ScoreResult, ...]
    """One per `ROUTE_GRID` weight; the same scored queries, in the same order, in each."""
    buckets: tuple[Bucket, ...]

    def __post_init__(self) -> None:
        ids = [tuple(q.query_id for q in r.per_query) for r in self.results]
        if len(set(ids)) != 1 or len(ids[0]) != len(self.buckets):
            msg = f"{self.set_id}: grid rankings score different queries"
            raise ValueError(msg)

    def at(self, weight_of: Mapping[Bucket, float]) -> ScoreResult:
        """The set scored with each query at its bucket's weight."""
        column = {w: i for i, w in enumerate(ROUTE_GRID)}
        per_query = tuple(
            self.results[column[weight_of[b]]].per_query[j] for j, b in enumerate(self.buckets)
        )
        return replace(self.results[0], per_query=per_query)


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
        kept = [q for q in queries if q.query_id in scored]
        grids.append(
            SetGrid(
                set_id=set_id,
                results=results,
                buckets=tuple(bucket_of(p_of[q.image_id], chroma[q.image_id]) for q in kept),
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
