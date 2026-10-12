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
    QueryScore,
    SearchQuery,
    bootstrap,
    paired_bootstrap,
    score,
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
    route=Route.MAC,
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


def test_ties_break_by_entity_id() -> None:
    run = vector_set({"a": [1, 0], "b": [1, 0], "q": [1, 0.01]})
    gallery = Gallery.from_vectors(run, {"a": "A", "b": "B"}, ["a", "b"])
    ranks = [score(run, gallery, [SearchQuery("q", "q", (e,))]).per_query[0].rank for e in "AB"]
    assert ranks == [1, 2]


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
