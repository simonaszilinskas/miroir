"""Score a search stack: one or more image models, fused (step G), maybe expanded (step E5).

Every model searches the same gallery list with the same queries through
`scoring.entity_scores`, the scorer `miroir eval run` uses. Their entity scores are fused
(`fusion.fuse`), ranked with `scoring.rank_entities`, and written as one result file. With
`not_found`, the same stack is also scored on T8 (`notfound`).

Fusion weights: give them, or let a small grid pick one first-model weight that does best on
all the scored sets together (mean hit@10 over sets, each set counting once), never one per
set. Only tune halves are ever searched here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime

import numpy as np

from miroir.config import Settings, eval_result_path
from miroir.eval.expansion import QE_TOP, expand_queries
from miroir.eval.fusion import Method, fuse, split_weight
from miroir.eval.notfound import Confidence, NotFoundScore, confidences, not_found_score
from miroir.eval.results import RunResult, set_score, write_result
from miroir.eval.run import publishable_only, search_queries
from miroir.eval.scoring import (
    Asked,
    EntityScores,
    Gallery,
    NotFoundQuery,
    ScoreResult,
    SearchQuery,
    entity_scores,
    per_query_values,
    rank_entities,
)
from miroir.eval.sets import Query, load_queries, read_gallery
from miroir.eval.vectors import Route, RunInfo, VectorSet, read_vectors
from miroir.records.store import read_images

logger = logging.getLogger(__name__)

#: First-model weights tried when no weight is given.
WEIGHT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
NOT_FOUND_SET = "T8"
#: Sets whose queries stand for searches that should match: T8 draws its queries from them.
IN_INDEX_SETS = ("T1", "T5")


@dataclass(frozen=True, slots=True)
class Stack:
    """Which models, how they are fused, and whether queries are expanded."""

    runs: tuple[str, ...]
    method: Method | None = None
    """Needed with two or more runs."""
    first_weight: float | None = None
    """Weight of the first run; the others share the rest. None with 2+ runs: pick from the grid."""
    qe_alpha: float | None = None
    """Expand queries with alpha-QE (E5) at this alpha. None: no expansion."""

    def __post_init__(self) -> None:
        if not self.runs or len(set(self.runs)) != len(self.runs):
            msg = f"a stack needs distinct runs, got {self.runs}"
            raise ValueError(msg)
        if (len(self.runs) > 1) != (self.method is not None):
            msg = "give a fusion method exactly when the stack has two or more runs"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Model:
    run: str
    vectors: VectorSet
    gallery: Gallery


@dataclass(frozen=True, slots=True)
class _Scored:
    """One set's queries with every model's entity scores on them."""

    queries: tuple[SearchQuery, ...]
    parts: tuple[EntityScores, ...]


def score_stack(
    settings: Settings,
    stack: Stack,
    *,
    round_id: str,
    step: str,
    option: str,
    set_ids: Sequence[str],
    gallery: str,
    publishable_queries: bool = False,
    not_found: bool = False,
) -> RunResult:
    """Score the stack on each set's tune half (and T8 if asked); write one result file."""
    gallery_ids = read_gallery(settings, gallery)
    entity_of = {i.image_id: i.entity_id for i in read_images(settings.images_parquet)}
    models = tuple(load_model(settings, run, entity_of, gallery_ids) for run in stack.runs)

    def queries_of(set_id: str) -> list[Query]:
        loaded = load_queries(settings, set_id, "tune")
        return (
            publishable_only(settings, loaded, in_gallery=gallery_ids)
            if (publishable_queries)
            else loaded
        )

    def scored_set(set_id: str) -> _Scored:
        queries = tuple(search_queries(queries_of(set_id)))
        return _Scored(queries, _score_models(models, queries, stack))

    scored = {s: scored_set(s) for s in set_ids}
    weights, chosen_note = _weights(stack, scored)
    scores = [
        set_score(
            rank_entities(_fused(s.parts, weights, stack), s.queries),
            test_set=set_id,
            half="tune",
            gallery=gallery,
        )
        for set_id, s in scored.items()
    ]
    found: tuple[NotFoundScore, ...] = ()
    if not_found:
        for set_id in IN_INDEX_SETS:
            if set_id not in scored:
                scored[set_id] = scored_set(set_id)
        missing = _score_models(models, not_found_queries(queries_of(NOT_FOUND_SET)), stack)
        found = (
            not_found_score(
                [c for s in IN_INDEX_SETS for c in _confidences(scored[s].parts, weights, stack)],
                _confidences(missing, weights, stack),
                test_set=NOT_FOUND_SET,
                half="tune",
                gallery=gallery,
                in_index_sets=IN_INDEX_SETS,
            ),
        )
    run_name = "+".join(stack.runs)
    return write_result(
        eval_result_path(round_id, step, option),
        round_id=round_id,
        step=step,
        option=option,
        vector_run=run_name,
        query_vector_run=run_name,
        vectors=_info(models, stack, weights, chosen_note),
        scores=scores,
        not_found=found,
        today=_today(),
    )


def load_model(settings: Settings, run: str, entity_of: dict[str, str], ids: list[str]) -> Model:
    vectors = read_vectors(settings.vectors_path(run))
    return Model(run, vectors, Gallery.from_vectors(vectors, entity_of, ids))


def _score_models(
    models: Sequence[Model], queries: Sequence[Asked], stack: Stack
) -> tuple[EntityScores, ...]:
    parts: list[EntityScores] = []
    for model in models:
        asking = model.vectors
        if stack.qe_alpha is not None:
            asking = expand_queries(asking, model.gallery, queries, alpha=stack.qe_alpha)
        parts.append(entity_scores(asking, model.gallery, queries))
    return tuple(parts)


def _fused(parts: Sequence[EntityScores], weights: Sequence[float], stack: Stack) -> EntityScores:
    if stack.method is None:
        return parts[0]
    return fuse(parts, weights, stack.method)


def _weights(stack: Stack, scored: dict[str, _Scored]) -> tuple[tuple[float, ...], str]:
    """The fusion weights, picked from the grid on all sets together when none is given."""
    count = len(stack.runs)
    if count == 1:
        return (1.0,), ""
    if stack.first_weight is not None:
        return split_weight(stack.first_weight, count), ""
    tried: dict[float, tuple[float, float]] = {}
    for first in WEIGHT_GRID:
        weights = split_weight(first, count)
        results = [
            rank_entities(_fused(s.parts, weights, stack), s.queries) for s in scored.values()
        ]
        tried[first] = mean_over_sets(results)
        logger.info("weight %.1f: mean hit@10 %.3f, mean mAP %.3f", first, *tried[first])
    # Best mean hit@10, then mAP; a tie goes to the weight nearest equal shares.
    nearest_even = sorted(WEIGHT_GRID, key=lambda f: abs(f - 1 / count))
    best = max(nearest_even, key=lambda f: tried[f])
    grid = ", ".join(f"{f:.1f}: {100 * h:.1f}" for f, (h, _) in tried.items())
    note = (
        f"first weight {best:.1f} picked from the grid by mean hit@10 over "
        f"{', '.join(scored)} (tune halves; {grid})"
    )
    return split_weight(best, count), note


def mean_over_sets(results: Sequence[ScoreResult]) -> tuple[float, float]:
    values = [per_query_values(r.per_query) for r in results]
    return (
        float(np.mean([v["hit@10"].mean() for v in values])),
        float(np.mean([v["map"].mean() for v in values])),
    )


def _confidences(
    parts: Sequence[EntityScores], weights: Sequence[float], stack: Stack
) -> tuple[Confidence, ...]:
    return confidences(_fused(parts, weights, stack), parts, weights)


def not_found_queries(queries: Sequence[Query]) -> list[NotFoundQuery]:
    with_answer = sum(1 for q in queries if q.expected_entity_ids)
    if with_answer:
        msg = f"{with_answer} {NOT_FOUND_SET} queries have a right answer"
        raise ValueError(msg)
    return [NotFoundQuery(q.query_id, q.query_image_id, q.exclude_image_ids) for q in queries]


def _info(
    models: Sequence[Model], stack: Stack, weights: Sequence[float], chosen_note: str
) -> RunInfo:
    infos = [m.vectors.info for m in models]
    notes: list[str] = []
    if stack.method is not None:
        parts = ", ".join(f"{i.option} {w:.2f}" for i, w in zip(infos, weights, strict=True))
        notes.append(f"{stack.method.value} fusion of {parts}")
        if chosen_note:
            notes.append(chosen_note)
    if stack.qe_alpha is not None:
        notes.append(f"E5 alpha-QE: query plus its top {QE_TOP} images, alpha {stack.qe_alpha}")
    if len(infos) == 1:
        first = infos[0]
        joined = "; ".join(n for n in (first.notes, *notes) if n)
        return replace(first, notes=joined)
    return combined_info(infos, "; ".join(notes))


def combined_info(infos: Sequence[RunInfo], notes: str) -> RunInfo:
    """One `RunInfo` for several models used together: costs and times add up."""
    routes = {i.route for i in infos}
    return RunInfo(
        option="+".join(i.option for i in infos),
        model=" + ".join(i.model for i in infos),
        resolution=max(i.resolution for i in infos),
        pooling="fusion",
        gallery=infos[0].gallery,
        created=max(i.created for i in infos),
        cost_usd=sum(i.cost_usd for i in infos),
        wall_time_s=sum(i.wall_time_s for i in infos),
        route=routes.pop() if len(routes) == 1 else Route.HOSTED,
        notes=notes,
    )


def _today() -> date:
    return datetime.now(UTC).date()
