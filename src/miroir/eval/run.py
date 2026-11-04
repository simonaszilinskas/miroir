"""What the `miroir eval` commands do: score a vector run on a test set, or compare."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from miroir.commons.images import commons_photos
from miroir.commons.model import View, is_commons_image
from miroir.config import Settings, eval_result_path
from miroir.eval.results import (
    SetScore,
    find_set_score,
    read_result,
    set_score,
    write_result,
)
from miroir.eval.scoring import (
    Difference,
    Gallery,
    Interval,
    NotFoundQuery,
    QueryScore,
    SearchQuery,
    bootstrap,
    hits,
    paired_bootstrap,
    score,
)
from miroir.eval.sets import Query, cluster_of, load_queries, read_gallery, write_gallery
from miroir.eval.vectors import Route, RunInfo, VectorSet, merge_runs, read_vectors
from miroir.records.store import read_commons, read_images, read_records

logger = logging.getLogger(__name__)


def score_test_set(
    settings: Settings,
    *,
    round_id: str,
    step: str,
    option: str,
    vector_run: str,
    set_id: str,
    query_vector_run: str | None = None,
    gallery: str | None = None,
    publishable_queries: bool = False,
) -> SetScore:
    """Score one test set's tune half and write it into eval/results/<round>/<step>-<option>.json.
    The sealed halves open in round R4 only, through `miroir.eval.r4`.

    `gallery` names a list in the test sets' galleries folder (e.g. "screening-20k"); every
    image in it needs a vector. Without it, the gallery is the whole vector run.
    `publishable_queries` keeps only queries whose image is publishable: the queries a closed
    model may see. Score open models with it too when comparing them with a closed one.
    """
    vectors = read_vectors(settings.vectors_path(vector_run))
    asking = (
        vectors
        if query_vector_run is None
        else read_vectors(settings.vectors_path(query_vector_run))
    )
    searched, queries = _search_inputs(
        settings,
        vectors,
        set_id=set_id,
        gallery=gallery,
        publishable_queries=publishable_queries,
    )
    scored = set_score(
        score(asking, searched, queries),
        test_set=set_id,
        half="tune",
        gallery=gallery or vectors.info.gallery,
    )
    write_result(
        eval_result_path(round_id, step, option),
        round_id=round_id,
        step=step,
        option=option,
        vector_run=vector_run,
        query_vector_run=query_vector_run or vector_run,
        vectors=vectors.info,
        scores=[scored],
        today=_today(),
    )
    return scored


def _search_inputs(
    settings: Settings,
    vectors: VectorSet,
    *,
    set_id: str,
    gallery: str | None,
    publishable_queries: bool,
) -> tuple[Gallery, list[SearchQuery]]:
    entity_of = entity_map(settings)
    ids = None if gallery is None else read_gallery(settings, gallery)
    searched = Gallery.from_vectors(vectors, entity_of, ids)
    loaded = load_queries(settings, set_id, "tune")
    if publishable_queries:
        loaded = publishable_only(settings, loaded, in_gallery=ids)
    return searched, search_queries(loaded)


def entity_map(settings: Settings) -> dict[str, str]:
    """The entity each gallery image shows: POP images, and Commons photos added to buildings
    (step F2, `commons-gallery-*.parquet`), which count as images of their building."""
    entity_of = {i.image_id: i.entity_id for i in read_images(settings.images_parquet)}
    for path in settings.commons_gallery_parquets():
        entity_of.update((p.image_id, p.entity_id) for p in read_commons(path))
    return entity_of


@dataclass(frozen=True, slots=True)
class Model:
    """One model's gallery run, and where its query vectors are."""

    run: str
    vectors: VectorSet
    gallery: Gallery
    asking: VectorSet
    """The query run, else the gallery run."""


def load_model(
    settings: Settings,
    run: str,
    entity_of: dict[str, str],
    ids: list[str],
    *,
    query_run: str | None = None,
) -> Model:
    """A model's gallery, and where its query vectors are (`query_run`, else the gallery run)."""
    vectors = read_vectors(settings.vectors_path(run))
    asking = vectors if query_run is None else read_vectors(settings.vectors_path(query_run))
    return Model(run, vectors, Gallery.from_vectors(vectors, entity_of, ids), asking)


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


def not_found_queries(queries: Sequence[Query], set_id: str) -> list[NotFoundQuery]:
    """A not-found set's queries (T8, T8c), each hiding its own entity's images."""
    with_answer = sum(1 for q in queries if q.expected_entity_ids)
    if with_answer:
        msg = f"{with_answer} {set_id} queries have a right answer"
        raise ValueError(msg)
    return [
        NotFoundQuery(q.query_id, q.query_image_id, q.exclude_image_ids, hidden_entity=q.entity_id)
        for q in queries
    ]


def compare(
    a: Path, b: Path, *, test_set: str, half: str, gallery: str, gallery_b: str | None = None
) -> tuple[Difference, ...]:
    """Paired bootstrap of run a minus run b on the same queries.

    `gallery_b` names run b's gallery when it differs from a's, e.g. a gallery with F2
    photos against the same gallery without them.
    """
    score_a = find_set_score(read_result(a), test_set, half, gallery)
    score_b = find_set_score(read_result(b), test_set, half, gallery_b or gallery)
    return paired_bootstrap(score_a.per_query, score_b.per_query)


def publishable_only(
    settings: Settings, queries: list[Query], *, in_gallery: Sequence[str] | None = None
) -> list[Query]:
    """Queries whose image comes from a publishable record (technical spec §7, closed models).

    Commons photos (T2, T2b) are openly licensed: they stay when an answer image is publishable.

    With `in_gallery`, also drop queries whose image is not in that list (images a service
    could not fetch are left out of the list, so every option answers the same queries).
    """
    records = read_records(settings.records_parquet, settings.images_parquet)
    public = {i.image_id for r in records if r.publishable for i in r.images}
    if in_gallery is not None:
        public &= set(in_gallery)
    public_entities = {r.entity_id for r in records for i in r.images if i.image_id in public}
    kept = [
        q
        for q in queries
        if q.query_image_id in public
        # A Commons photo is openly licensed; it needs an answer a closed model may see, or
        # none at all (T8c, where the right answer is "not found").
        or (
            is_commons_image(q.query_image_id)
            and (not q.expected_entity_ids or set(q.expected_entity_ids) & public_entities)
        )
    ]
    logger.info("kept %d of %d queries with a publishable image", len(kept), len(queries))
    return kept


def merge_vector_runs(settings: Settings, runs: Sequence[str], *, out: str, gallery: str) -> int:
    """Write run `out` holding the vectors of `runs` (one model); returns its row count."""
    loaded = [read_vectors(settings.vectors_path(r)) for r in runs]
    return len(merge_runs(settings.vectors_path(out), loaded, gallery=gallery).ids)


def write_fetched_gallery(settings: Settings, gallery: str, runs: Sequence[str]) -> list[str]:
    """Write `<gallery>-fetched`: the images of `gallery` that every run holds a vector for.

    A service skips images it cannot fetch; scoring every run on this list keeps the gallery
    the same for all of them. Returns the list, in the gallery's order.
    """
    if not runs:
        msg = "give the runs to compare"
        raise ValueError(msg)
    held = [set(read_vectors(settings.vectors_path(r)).ids) for r in runs]
    ids = [i for i in read_gallery(settings, gallery) if all(i in h for h in held)]
    write_gallery(settings, f"{gallery}-fetched", ids)
    return ids


@dataclass(frozen=True, slots=True)
class ViewScore:
    """One option's scores on the queries of a Commons set that show one view."""

    view: View
    queries: int
    hit_at_1: Interval
    hit_at_10: Interval


def score_by_view(
    settings: Settings, result: Path, *, test_set: str, gallery: str
) -> tuple[ViewScore, ...]:
    """Split a scored tune half of a Commons set (T2) by the view each photo shows.

    Views (exterior, interior, detail) are guessed from file words when the set is built.
    """
    scored = find_set_score(read_result(result), test_set, "tune", gallery)
    image_of = {q.query_id: q.query_image_id for q in load_queries(settings, test_set, "tune")}
    view_of = {i: p.view for i, p in commons_photos(settings).items()}
    by_view: dict[View, list[QueryScore]] = {}
    for q in scored.per_query:
        by_view.setdefault(view_of[image_of[q.query_id]], []).append(q)
    out: list[ViewScore] = []
    for view, scores in sorted(by_view.items()):
        ranks = [q.rank for q in scores]
        clusters = [q.cluster for q in scores]
        out.append(
            ViewScore(
                view,
                len(ranks),
                bootstrap(hits(ranks, 1), clusters=clusters),
                bootstrap(hits(ranks, 10), clusters=clusters),
            )
        )
    return tuple(out)


def search_queries(queries: list[Query]) -> list[SearchQuery]:
    """Test set queries as the scorer takes them. T8 queries (no right answer) are refused."""
    not_found = sum(1 for q in queries if not q.expected_entity_ids)
    if not_found:
        msg = (
            f"{not_found} queries have no right answer (T8): score them with the not-found measure"
        )
        raise ValueError(msg)
    return [
        SearchQuery(
            query_id=q.query_id,
            image_id=q.query_image_id,
            entity_ids=q.expected_entity_ids,
            excluded_image_ids=q.exclude_image_ids,
            cluster=cluster_of(q),
        )
        for q in queries
    ]


def _today() -> date:
    return datetime.now(UTC).date()
