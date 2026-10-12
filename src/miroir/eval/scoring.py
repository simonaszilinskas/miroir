"""Exact cosine search and scoring (experiments.md §4).

Search is flat: every query is compared with every gallery image (NumPy, no approximate index).
Images are grouped by entity: an entity scores the best of its images, and the ranking is a
ranking of entities. A query is right at rank r when one of its right entities is r-th.

Ties between entities with the same score are broken by entity id order, so results never
depend on row order and never flatter an option by counting a tie as a win.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from miroir.config import SEED
from miroir.eval.vectors import VectorSet

logger = logging.getLogger(__name__)

#: Hit rates reported for every run.
HIT_KS = (1, 5, 10)
#: Bootstrap resamples for every interval.
RESAMPLES = 1_000

_QUERY_CHUNK = 256

Floats = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """One test query: the image that asks, the entities that count as right, what to hide.

    The query image itself is always hidden from the gallery. `excluded_image_ids` adds the
    rest, e.g. every other image of the same photo record (technical spec §7).
    """

    query_id: str
    image_id: str
    entity_ids: tuple[str, ...]
    excluded_image_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entity_ids:
            msg = f"query {self.query_id} has no right entity"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Gallery:
    """The images searched, with the entity each one shows. Any subset of a vector run."""

    image_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    vectors: NDArray[np.float16]

    def __post_init__(self) -> None:
        if not (len(self.image_ids) == len(self.entity_ids) == self.vectors.shape[0]):
            msg = "gallery ids, entities and vectors differ in length"
            raise ValueError(msg)
        if len(set(self.image_ids)) != len(self.image_ids):
            msg = "gallery holds duplicate image ids"
            raise ValueError(msg)

    @classmethod
    def from_vectors(
        cls,
        run: VectorSet,
        entity_of: Mapping[str, str],
        image_ids: Sequence[str] | None = None,
    ) -> Gallery:
        """Gallery of `image_ids` (default: every id of the run). Each needs a vector and entity."""
        wanted = tuple(run.ids) if image_ids is None else tuple(image_ids)
        row_of = run.index()
        missing_vector = [i for i in wanted if i not in row_of]
        missing_entity = [i for i in wanted if i not in entity_of]
        if missing_vector or missing_entity:
            msg = (
                f"gallery: {len(missing_vector)} image(s) without a vector "
                f"(e.g. {missing_vector[:3]}), {len(missing_entity)} without an entity "
                f"(e.g. {missing_entity[:3]})"
            )
            raise ValueError(msg)
        rows = np.fromiter((row_of[i] for i in wanted), dtype=np.int64, count=len(wanted))
        return cls(
            image_ids=wanted,
            entity_ids=tuple(entity_of[i] for i in wanted),
            vectors=np.asarray(run.vectors[rows]),
        )

    @property
    def entity_count(self) -> int:
        return len(set(self.entity_ids))


@dataclass(frozen=True, slots=True)
class QueryScore:
    query_id: str
    rank: int
    """Rank of the best-placed right entity, from 1."""
    ap: float
    """Average precision over the right entities present in the gallery."""


@dataclass(frozen=True, slots=True)
class ScoreResult:
    per_query: tuple[QueryScore, ...]
    skipped_no_answer: int
    """Queries left out because none of their right entities is in the gallery."""
    gallery_images: int
    gallery_entities: int
    ap_max_rank: int | None


@dataclass(frozen=True, slots=True)
class Interval:
    """A mean with its 95% bootstrap interval."""

    value: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class Difference:
    """Paired difference `a - b` on the same queries, with its 95% bootstrap interval."""

    metric: str
    a: float
    b: float
    diff: Interval
    share_not_better: float
    """Share of resamples where a does not beat b. Under 0.025: a wins beyond the error bar."""


def score(
    query_vectors: VectorSet,
    gallery: Gallery,
    queries: Sequence[SearchQuery],
    *,
    ap_max_rank: int | None = None,
) -> ScoreResult:
    """Search every query against the whole gallery and rank entities.

    `ap_max_rank` counts right entities past that rank as not found in AP, as benchmarks with
    a top-k cut do. None ranks the whole gallery.
    """
    if not queries:
        msg = "no queries to score"
        raise ValueError(msg)
    if len({q.query_id for q in queries}) != len(queries):
        msg = "query ids are not unique"
        raise ValueError(msg)
    if query_vectors.dim != gallery.vectors.shape[1]:
        msg = f"query dim {query_vectors.dim} != gallery dim {gallery.vectors.shape[1]}"
        raise ValueError(msg)
    query_row = query_vectors.index()
    no_vector = [q.image_id for q in queries if q.image_id not in query_row]
    if no_vector:
        msg = f"{len(no_vector)} query image(s) have no vector, e.g. {no_vector[:3]}"
        raise ValueError(msg)

    entity_names, codes = np.unique(np.asarray(gallery.entity_ids), return_inverse=True)
    code_of = {str(name): i for i, name in enumerate(entity_names)}
    order = np.argsort(codes, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(codes[order]) != 0])
    gallery_row = {image_id: i for i, image_id in enumerate(gallery.image_ids)}
    matrix: Floats = np.asarray(gallery.vectors, dtype=np.float32)

    results: list[QueryScore] = []
    skipped = 0
    for start in range(0, len(queries), _QUERY_CHUNK):
        chunk = queries[start : start + _QUERY_CHUNK]
        rows = [query_row[q.image_id] for q in chunk]
        asked: Floats = np.asarray(query_vectors.vectors[rows], dtype=np.float32)
        sims = asked @ matrix.T
        for j, query in enumerate(chunk):
            hidden = [
                gallery_row[i]
                for i in (query.image_id, *query.excluded_image_ids)
                if i in gallery_row
            ]
            sims[j, hidden] = -np.inf
        entity_scores = np.maximum.reduceat(sims[:, order], starts, axis=1)
        for j, query in enumerate(chunk):
            scored = _score_one(query, entity_scores[j], code_of, ap_max_rank)
            if scored is None:
                skipped += 1
            else:
                results.append(scored)
    if skipped:
        logger.warning("skipped %d of %d queries: no answer in gallery", skipped, len(queries))
    return ScoreResult(
        per_query=tuple(results),
        skipped_no_answer=skipped,
        gallery_images=len(gallery.image_ids),
        gallery_entities=len(entity_names),
        ap_max_rank=ap_max_rank,
    )


def _score_one(
    query: SearchQuery, scores: Floats, code_of: Mapping[str, int], ap_max_rank: int | None
) -> QueryScore | None:
    right = [code_of[e] for e in dict.fromkeys(query.entity_ids) if e in code_of]
    right = [c for c in right if np.isfinite(scores[c])]
    if not right:
        return None
    ranks = sorted(
        int((scores > scores[c]).sum() + (scores[:c] == scores[c]).sum()) + 1 for c in right
    )
    cut = ap_max_rank if ap_max_rank is not None else len(scores)
    ap = sum((i + 1) / r for i, r in enumerate(ranks) if r <= cut) / len(ranks)
    return QueryScore(query_id=query.query_id, rank=ranks[0], ap=ap)


def hits(result_ranks: Sequence[int], k: int) -> NDArray[np.float64]:
    return (np.asarray(result_ranks) <= k).astype(np.float64)


def per_query_values(per_query: Sequence[QueryScore]) -> dict[str, NDArray[np.float64]]:
    """Each metric's value per query, keyed by metric name ("hit@1", ..., "map")."""
    ranks = [q.rank for q in per_query]
    values = {f"hit@{k}": hits(ranks, k) for k in HIT_KS}
    values["map"] = np.asarray([q.ap for q in per_query], dtype=np.float64)
    return values


def bootstrap(values: NDArray[np.float64], *, seed: int = SEED) -> Interval:
    """Mean and 95% percentile interval over RESAMPLES resamples of the queries."""
    means = _resampled_means(values, seed)
    low, high = np.percentile(means, [2.5, 97.5])
    return Interval(value=float(values.mean()), low=float(low), high=float(high))


def paired_bootstrap(
    a: Sequence[QueryScore], b: Sequence[QueryScore], *, seed: int = SEED
) -> tuple[Difference, ...]:
    """Difference a - b for every metric, resampling the same queries for both runs."""
    a_by_id = {q.query_id: q for q in a}
    b_by_id = {q.query_id: q for q in b}
    if a_by_id.keys() != b_by_id.keys():
        only_a = len(a_by_id.keys() - b_by_id.keys())
        only_b = len(b_by_id.keys() - a_by_id.keys())
        msg = f"runs answer different queries: {only_a} only in a, {only_b} only in b"
        raise ValueError(msg)
    ids = sorted(a_by_id)
    a_values = per_query_values([a_by_id[i] for i in ids])
    b_values = per_query_values([b_by_id[i] for i in ids])
    out: list[Difference] = []
    for metric, a_metric in a_values.items():
        diff = a_metric - b_values[metric]
        means = _resampled_means(diff, seed)
        low, high = np.percentile(means, [2.5, 97.5])
        out.append(
            Difference(
                metric=metric,
                a=float(a_metric.mean()),
                b=float(b_values[metric].mean()),
                diff=Interval(value=float(diff.mean()), low=float(low), high=float(high)),
                share_not_better=float((means <= 0).mean()),
            )
        )
    return tuple(out)


def _resampled_means(values: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
    if values.size == 0:
        msg = "cannot bootstrap zero queries"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(RESAMPLES, values.size))
    return values[picks].mean(axis=1)
