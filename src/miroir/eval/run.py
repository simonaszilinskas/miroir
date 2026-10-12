"""What the `miroir eval` commands do: score a vector run on a test set, a benchmark, or compare."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

from miroir.config import Half, Settings, eval_result_path
from miroir.eval import amstertime
from miroir.eval.results import (
    RunResult,
    SetScore,
    find_set_score,
    read_result,
    set_score,
    write_result,
)
from miroir.eval.scoring import Difference, Gallery, SearchQuery, paired_bootstrap, score
from miroir.eval.sets import Query, load_queries, read_gallery
from miroir.eval.vectors import read_vectors
from miroir.records.store import read_images

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
) -> SetScore:
    """Score one test set half and write it into eval/results/<round>/<step>-<option>.json.

    `gallery` names a list in the test sets' galleries folder (e.g. "screening-20k"); every
    image in it needs a vector. Without it, the gallery is the whole vector run.
    """
    vectors = read_vectors(settings.vectors_path(vector_run))
    asking = (
        vectors
        if query_vector_run is None
        else read_vectors(settings.vectors_path(query_vector_run))
    )
    entity_of = {i.image_id: i.entity_id for i in read_images(settings.images_parquet)}
    ids = None if gallery is None else read_gallery(settings, gallery)
    searched = Gallery.from_vectors(vectors, entity_of, ids)
    queries = search_queries(load_queries(settings, set_id, half, allow_sealed=allow_sealed))
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


def compare(a: Path, b: Path, *, test_set: str, half: str, gallery: str) -> tuple[Difference, ...]:
    """Paired bootstrap of run a minus run b on the same queries."""
    score_a = find_set_score(read_result(a), test_set, half, gallery)
    score_b = find_set_score(read_result(b), test_set, half, gallery)
    return paired_bootstrap(score_a.per_query, score_b.per_query)


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
