from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from miroir.eval.results import (
    SUMMARY_END,
    SUMMARY_START,
    ResultFormatError,
    read_result,
    rebuild_summary,
    set_score,
    write_result,
)
from miroir.eval.scoring import (
    Gallery,
    NotFoundQuery,
    QueryScore,
    SearchQuery,
    bootstrap,
    entity_scores,
    first_rank,
    paired_bootstrap,
    score,
    tie_ranks,
)
from miroir.eval.vectors import Route, RunInfo, VectorSet

INFO = RunInfo(
    option="A1",
    model="test/model",
    resolution=224,
    pooling="cls",
    gallery="fixture",
    created=date(2026, 10, 12),
    cost_usd=0.0,
    wall_time_s=0.0,
    route=Route.HOSTED,
)


def vector_set(rows: dict[str, list[float]]) -> VectorSet:
    matrix = np.asarray(list(rows.values()), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return VectorSet(ids=tuple(rows), vectors=matrix.astype(np.float16), info=INFO)


# Gallery on the unit circle. Query [0, 1] scores: g1 0, g2 0.6, g3 1, g4 0.8.
# Entities: X = {g1, g4} scores 0.8, Y = {g2} 0.6, Z = {g3} 1. Ranking: Z, X, Y.
GALLERY_RUN = vector_set(
    {"g1": [1, 0], "g2": [0.8, 0.6], "g3": [0, 1], "g4": [0.6, 0.8], "q": [0, 1]}
)
ENTITY = {"g1": "X", "g2": "Y", "g3": "Z", "g4": "X", "q": "Q"}
GALLERY = Gallery.from_vectors(GALLERY_RUN, ENTITY, ["g1", "g2", "g3", "g4"])


def one(query: SearchQuery, **kwargs: int) -> QueryScore:
    result = score(GALLERY_RUN, GALLERY, [query], **kwargs)
    assert result.skipped_no_answer == 0
    return result.per_query[0]


def test_entity_scores_its_best_image() -> None:
    hit = one(SearchQuery("q1", "q", ("X",)))
    assert (hit.rank, hit.ap) == (2, 0.5)


def test_ap_over_several_right_entities() -> None:
    hit = one(SearchQuery("q1", "q", ("X", "Y")))
    assert hit.rank == 2
    assert hit.ap == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_excluded_images_leave_the_gallery() -> None:
    hit = one(SearchQuery("q1", "q", ("X",), excluded_image_ids=("g4",)))
    assert hit.rank == 3


def test_a_not_found_query_hides_every_image_of_its_entity() -> None:
    # g4 stands for a Commons photo added to X after the set was built (step F2): the
    # query's exclusions do not name it, its entity does.
    asked = NotFoundQuery("t8", "q", excluded_image_ids=("g1",), hidden_entity="X")
    found = entity_scores(GALLERY_RUN, GALLERY, [asked])
    assert found.values[0][found.entity_ids.index("X")] == -np.inf
    assert np.isfinite(found.values[0]).sum() == 2  # Y and Z are still searched


def test_query_image_never_finds_itself() -> None:
    run = vector_set({"a": [1, 0], "b": [0.9, 0.1], "c": [0, 1]})
    gallery = Gallery.from_vectors(run, {"a": "A", "b": "B", "c": "C"})
    result = score(run, gallery, [SearchQuery("a", "a", ("B",))])
    assert result.per_query[0].rank == 1


def test_query_without_answer_in_gallery_is_skipped() -> None:
    result = score(GALLERY_RUN, GALLERY, [SearchQuery("q1", "q", ("nowhere",))])
    assert result.per_query == ()
    assert result.skipped_no_answer == 1


def test_ap_cut_at_max_rank() -> None:
    hit = one(SearchQuery("q1", "q", ("X",)), ap_max_rank=1)
    assert (hit.rank, hit.ap) == (2, 0.0)


def test_a_tie_with_a_wrong_entity_counts_against_the_query() -> None:
    run = vector_set({"a": [1, 0], "b": [1, 0], "q": [1, 0.01]})
    gallery = Gallery.from_vectors(run, {"a": "A", "b": "B"}, ["a", "b"])
    ranks = [score(run, gallery, [SearchQuery("q", "q", (e,))]).per_query[0].rank for e in "AB"]
    assert ranks == [2, 2]  # whichever is right, the tied wrong one goes first


def test_right_entities_tied_together_do_not_push_each_other_down() -> None:
    run = vector_set({"a": [1, 0], "b": [1, 0], "c": [0, 1], "q": [1, 0.01]})
    gallery = Gallery.from_vectors(run, {"a": "A", "b": "B", "c": "C"}, ["a", "b", "c"])
    hit = score(run, gallery, [SearchQuery("q", "q", ("A", "B"))]).per_query[0]
    assert (hit.rank, hit.ap) == (1, 1.0)


def test_tie_ranks_put_every_tied_wrong_entity_first() -> None:
    scores = np.asarray([0.5, 0.9, 0.9, 0.9, -np.inf], dtype=np.float32)
    assert tie_ranks(scores, [2]) == [3]
    assert tie_ranks(scores, [2, 0]) == [3, 4]
    assert tie_ranks(scores, [2, 1]) == [2, 3]
    assert first_rank(scores, [4]) == 6  # off the ranking: past every entity
    assert first_rank(scores, []) == 0


def test_rejects_query_without_vector() -> None:
    with pytest.raises(ValueError, match="no vector"):
        score(GALLERY_RUN, GALLERY, [SearchQuery("q1", "missing", ("X",))])


def test_gallery_needs_an_entity_per_image() -> None:
    with pytest.raises(ValueError, match="without an entity"):
        Gallery.from_vectors(GALLERY_RUN, {"g1": "X"}, ["g1", "g2"])


def test_bootstrap_of_constant_values_is_a_point() -> None:
    interval = bootstrap(np.ones(50))
    assert (interval.value, interval.low, interval.high) == (1.0, 1.0, 1.0)


def test_bootstrap_interval_has_the_expected_width() -> None:
    values = (np.arange(1000) % 2).astype(np.float64)  # mean 0.5, sd 0.5
    interval = bootstrap(values)
    assert interval.value == 0.5
    half_width = 1.96 * 0.5 / np.sqrt(1000)  # about 0.031
    assert interval.high - interval.value == pytest.approx(half_width, abs=0.006)
    assert interval.value - interval.low == pytest.approx(half_width, abs=0.006)


def test_bootstrap_is_seeded() -> None:
    values = np.random.default_rng(1).random(200)
    assert bootstrap(values) == bootstrap(values)


def test_bootstrap_resamples_clusters_not_queries() -> None:
    # 20 buildings, 10 photos each, all right or all wrong: 20 draws, not 200.
    values = np.repeat((np.arange(20) % 2).astype(np.float64), 10)
    clusters = [f"PA{i // 10}" for i in range(200)]
    iid = bootstrap(values)
    clustered = bootstrap(values, clusters=clusters)
    assert clustered.value == iid.value == 0.5
    half_width = 1.96 * 0.5 / np.sqrt(20)  # about 0.22, against 0.07 query by query
    assert clustered.high - clustered.value == pytest.approx(half_width, abs=0.05)
    assert iid.high - iid.value < 0.1


def test_one_query_per_cluster_bootstraps_as_before() -> None:
    values = np.random.default_rng(3).random(100)
    assert bootstrap(values, clusters=[f"e{i}" for i in range(100)]) == bootstrap(values)


def test_paired_bootstrap_resamples_the_clusters_of_the_scores() -> None:
    # Two buildings: on one, a wins all ten photos; on the other, it ties.
    better = [QueryScore(f"q{i}", 1, 1.0, cluster=f"PA{i // 10}") for i in range(20)]
    worse = [
        QueryScore(f"q{i}", 1 if i >= 10 else 20, 1.0, cluster=f"PA{i // 10}") for i in range(20)
    ]
    by_query = {d.metric: d for d in paired_bootstrap(_scores_of(better), _scores_of(worse))}
    by_building = {d.metric: d for d in paired_bootstrap(better, worse)}
    assert by_building["hit@1"].diff.value == by_query["hit@1"].diff.value == 0.5
    assert by_building["hit@1"].share_not_better > by_query["hit@1"].share_not_better


def _scores_of(scores: list[QueryScore]) -> list[QueryScore]:
    """The same scores with no cluster: each query its own."""
    return [QueryScore(q.query_id, q.rank, q.ap) for q in scores]


def test_a_query_score_reads_its_cluster_from_the_query() -> None:
    hit = one(SearchQuery("q1", "q", ("X",), cluster="PA1"))
    assert hit.cluster == "PA1"


def _scores(ranks: list[int]) -> list[QueryScore]:
    return [QueryScore(f"q{i}", r, 1 / r) for i, r in enumerate(ranks)]


def test_paired_bootstrap_sees_a_clear_win() -> None:
    better = _scores([1] * 100)
    worse = _scores([1] * 50 + [20] * 50)
    diff = {d.metric: d for d in paired_bootstrap(better, worse)}
    assert diff["hit@1"].diff.value == pytest.approx(0.5)
    assert diff["hit@1"].diff.low > 0.3
    assert diff["hit@1"].share_not_better == 0.0
    assert diff["hit@10"].diff.value == pytest.approx(0.5)


def test_paired_bootstrap_of_a_run_with_itself_is_zero() -> None:
    run = _scores([1, 2, 30, 4])
    for d in paired_bootstrap(run, run):
        assert (d.diff.low, d.diff.value, d.diff.high) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_needs_the_same_queries() -> None:
    with pytest.raises(ValueError, match="different queries"):
        paired_bootstrap(_scores([1, 2]), _scores([1]))


def _result_file(tmp_path: Path, test_set: str = "T1") -> Path:
    path = tmp_path / "results" / "R1" / "A-test.json"
    result = score(
        GALLERY_RUN, GALLERY, [SearchQuery("q1", "q", ("X",)), SearchQuery("q2", "q", ("Z",))]
    )
    write_result(
        path,
        round_id="R1",
        step="A",
        option="test",
        vector_run="run1",
        query_vector_run="run1",
        vectors=INFO,
        scores=[set_score(result, test_set=test_set, half="tune", gallery="fixture")],
        today=date(2026, 10, 12),
    )
    return path


def test_result_round_trip(tmp_path: Path) -> None:
    result = read_result(_result_file(tmp_path))
    (only,) = result.scores
    assert only.hit_at_1.value == 0.5
    assert only.hit_at_5.value == 1.0
    assert only.map.value == pytest.approx(0.75)
    assert [q.rank for q in only.per_query] == [2, 1]


def test_result_merges_by_set(tmp_path: Path) -> None:
    _result_file(tmp_path, "T1")
    path = _result_file(tmp_path, "T5")
    _result_file(tmp_path, "T5")
    assert [s.test_set for s in read_result(path).scores] == ["T1", "T5"]


def test_result_rejects_a_mean_that_does_not_match(tmp_path: Path) -> None:
    path = _result_file(tmp_path)
    path.write_text(path.read_text().replace('"value": 0.5', '"value": 0.9', 1))
    with pytest.raises(ResultFormatError, match="per-query mean"):
        read_result(path)


def test_summary_is_rebuilt_in_place(tmp_path: Path) -> None:
    _result_file(tmp_path)
    md = tmp_path / "results.md"
    md.write_text(f"# Results\n\n{SUMMARY_START}\nold table\n{SUMMARY_END}\n\nNotes stay.\n")
    assert rebuild_summary(tmp_path / "results", md) == 1
    text = md.read_text()
    assert "old table" not in text
    assert "Notes stay." in text
    assert "| R1 | T1 | tune | fixture | A-test | test/model | 224 | cls | 2 | 50.0 [" in text
