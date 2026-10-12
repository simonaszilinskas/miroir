"""What the `miroir eval` commands do: score a vector run on a test set, a benchmark, or compare."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

from miroir.commons.images import commons_photos
from miroir.commons.model import View, is_commons_image
from miroir.config import Half, Settings, eval_result_path
from miroir.eval import amstertime
from miroir.eval.rerank import MIX_INLIERS, SHORTLIST, Rerank, read_matches, rerank, search_times
from miroir.eval.results import (
    RunResult,
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
    ScoreResult,
    SearchQuery,
    Shortlist,
    bootstrap,
    hits,
    paired_bootstrap,
    score,
    shortlists,
)
from miroir.eval.sets import Query, load_queries, read_gallery, write_gallery
from miroir.eval.vectors import VectorSet, merge_runs, read_vectors
from miroir.jobs.spend import SpendKind, SpendLedger
from miroir.records.store import read_commons, read_images, read_records

logger = logging.getLogger(__name__)

AMSTERTIME_SET = amstertime.GALLERY


def score_test_set(
    settings: Settings,
    *,
    round_id: str,
    step: str,
    option: str,
    vector_run: str,
    set_id: str,
    half: Half,
    query_vector_run: str | None = None,
    gallery: str | None = None,
    allow_sealed: bool = False,
    publishable_queries: bool = False,
) -> SetScore:
    """Score one test set half and write it into eval/results/<round>/<step>-<option>.json.

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
        half=half,
        gallery=gallery,
        allow_sealed=allow_sealed,
        publishable_queries=publishable_queries,
    )
    scored = set_score(
        score(asking, searched, queries),
        test_set=set_id,
        half=half,
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
    half: Half,
    gallery: str | None,
    allow_sealed: bool,
    publishable_queries: bool,
) -> tuple[Gallery, list[SearchQuery]]:
    entity_of = entity_map(settings)
    ids = None if gallery is None else read_gallery(settings, gallery)
    searched = Gallery.from_vectors(vectors, entity_of, ids)
    loaded = load_queries(settings, set_id, half, allow_sealed=allow_sealed)
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


def rerank_shortlists(
    settings: Settings, *, vector_run: str, set_id: str, gallery: str, top: int = SHORTLIST
) -> tuple[tuple[Shortlist, ...], ScoreResult, VectorSet]:
    """Each tune query's top entities (step E1), on the queries a closed model may see.

    Also returns the plain search score of the same queries and the vectors searched.
    """
    vectors = read_vectors(settings.vectors_path(vector_run))
    searched, queries = _search_inputs(
        settings,
        vectors,
        set_id=set_id,
        half="tune",
        gallery=gallery,
        allow_sealed=False,
        publishable_queries=True,
    )
    lists, plain = shortlists(vectors, searched, queries, top=top)
    return lists, plain, vectors


def matches_name(vector_run: str, set_id: str, top: int = SHORTLIST) -> str:
    """Name of the file holding a shortlist's keypoint matches."""
    return f"E1-{vector_run}-{set_id}-top{top}"


def score_rerank(
    settings: Settings,
    *,
    round_id: str,
    label: str,
    vector_run: str,
    set_id: str,
    gallery: str,
    top: int = SHORTLIST,
) -> list[SetScore]:
    """Score a shortlist re-ranked each way (`Rerank`), one result file per way.

    Writes eval/results/<round>/E-E1-<label>-<way>.json. Matches must be on file
    (`miroir rerank modal`).
    """
    lists, plain, vectors = rerank_shortlists(
        settings, vector_run=vector_run, set_id=set_id, gallery=gallery, top=top
    )
    name = matches_name(vector_run, set_id, top)
    matched = read_matches(settings.matches_path(name))
    unmatched = [s.query_id for s in lists if s.query_id not in matched]
    if unmatched:
        msg = f"{name}: {len(unmatched)} queries have no matches yet, e.g. {unmatched[0]}"
        raise ValueError(msg)
    mine = [matched[s.query_id] for s in lists]
    median_s, p95_s = search_times(mine)
    cost = SpendLedger(settings.spend_ledger).total(SpendKind.RUN, run=name)
    out: list[SetScore] = []
    for how in Rerank:
        per_query = tuple(rerank(s, matched[s.query_id], how) for s in lists)
        result = replace(plain, per_query=per_query)
        scored = set_score(result, test_set=set_id, half="tune", gallery=gallery)
        info = replace(
            vectors.info,
            option="E1",
            model=f"{vectors.info.model} + ALIKED/LightGlue",
            cost_usd=cost,
            wall_time_s=sum(m.search_s + m.candidates_s for m in mine),
            notes=(
                f"top {top} entities re-ranked by {how} (homography inliers"
                f"{f', + cosine, inliers / {MIX_INLIERS:.0f}' if how is Rerank.MIX else ''}); "
                f"adds {median_s * 1000:.0f} ms median, {p95_s * 1000:.0f} ms p95 per search "
                "on a Modal L4 (gallery keypoints computed ahead)"
            ),
        )
        write_result(
            eval_result_path(round_id, "E", f"E1-{label}-{how}"),
            round_id=round_id,
            step="E",
            option=f"E1-{label}-{how}",
            vector_run=vector_run,
            query_vector_run=vector_run,
            vectors=info,
            scores=[scored],
            today=_today(),
        )
        out.append(scored)
    return out


def score_amstertime(
    settings: Settings, *, vector_run: str, option: str, round_id: str = "R0", step: str = "A"
) -> RunResult:
    """Score a vector run of AmsterTime in all three directions and write one result file."""
    vectors = read_vectors(settings.vectors_path(vector_run))
    amstertime.check_ids(vectors.ids)
    scores = [
        set_score(
            amstertime.score_amstertime(vectors, direction),
            test_set=f"{AMSTERTIME_SET}-{direction.value}",
            half="all",
            gallery=AMSTERTIME_SET,
        )
        for direction in amstertime.Direction
    ]
    return write_result(
        eval_result_path(round_id, step, option),
        round_id=round_id,
        step=step,
        option=option,
        vector_run=vector_run,
        query_vector_run=vector_run,
        vectors=vectors.info,
        scores=scores,
        today=_today(),
    )


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
    by_view: dict[View, list[int]] = {}
    for q in scored.per_query:
        by_view.setdefault(view_of[image_of[q.query_id]], []).append(q.rank)
    return tuple(
        ViewScore(view, len(ranks), bootstrap(hits(ranks, 1)), bootstrap(hits(ranks, 10)))
        for view, ranks in sorted(by_view.items())
    )


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
        )
        for q in queries
    ]


def _today() -> date:
    return datetime.now(UTC).date()
