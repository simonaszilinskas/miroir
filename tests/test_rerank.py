"""Step E1 on the Mac: shortlists from a vector search, and re-ranking them by inliers."""

from __future__ import annotations

from pathlib import Path

import pytest

from miroir.eval.rerank import (
    CandidateMatch,
    QueryMatches,
    Rerank,
    append_matches,
    read_matches,
    rerank,
    search_times,
)
from miroir.eval.scoring import SearchQuery, score, shortlists
from tests.test_scoring import GALLERY, GALLERY_RUN

# Query [0, 1]: Z (g3, 1.0), then X (best image g4, 0.8), then Y (g2, 0.6).
QUERY = SearchQuery("q1", "q", ("Y", "X"))


def matched(inliers: dict[str, int]) -> QueryMatches:
    found = tuple(CandidateMatch(i, n + 5, n, n) for i, n in inliers.items())
    return QueryMatches("q1", "q", found, query_s=0.01, match_s=0.1, candidates_s=0.5)


def test_shortlist_ranks_entities_with_their_best_image() -> None:
    (only,), plain = shortlists(GALLERY_RUN, GALLERY, [QUERY], top=2)
    assert only.entity_ids == ("Z", "X")
    assert only.image_ids == ("g3", "g4")
    assert only.scores == pytest.approx((1.0, 0.8), abs=1e-3)
    assert dict(only.right_ranks) == {"Y": 3, "X": 2}  # Y is past the shortlist
    assert plain.per_query == score(GALLERY_RUN, GALLERY, [QUERY]).per_query


def test_no_inliers_keeps_the_vector_order() -> None:
    (only,), plain = shortlists(GALLERY_RUN, GALLERY, [QUERY], top=3)
    flat = matched({"g3": 0, "g4": 0, "g2": 0})
    for how in Rerank:
        assert rerank(only, flat, how) == plain.per_query[0]


def test_inliers_lift_the_right_entity() -> None:
    (only,), _ = shortlists(GALLERY_RUN, GALLERY, [QUERY], top=3)
    found = rerank(only, matched({"g3": 10, "g4": 3, "g2": 80}), Rerank.INLIERS)
    assert found.rank == 1  # Y
    assert found.ap == pytest.approx((1 / 1 + 2 / 3) / 2)  # Y first, X third


def test_mix_weighs_cosine_against_inliers() -> None:
    (only,), _ = shortlists(GALLERY_RUN, GALLERY, [QUERY], top=3)
    # Y: 0.6 + 0.3 = 0.9 passes X (0.8) but not Z (1.0).
    assert rerank(only, matched({"g3": 0, "g4": 0, "g2": 30}), Rerank.MIX).rank == 2


def test_entities_past_the_shortlist_keep_their_rank() -> None:
    (only,), _ = shortlists(GALLERY_RUN, GALLERY, [QUERY], top=1)
    found = rerank(only, matched({"g3": 0}), Rerank.INLIERS)
    assert found.rank == 2  # X, untouched


def test_a_candidate_without_matches_is_refused() -> None:
    (only,), _ = shortlists(GALLERY_RUN, GALLERY, [QUERY], top=2)
    with pytest.raises(ValueError, match="no matches"):
        rerank(only, matched({"g3": 1}), Rerank.INLIERS)


def test_matches_round_trip_and_times(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    assert read_matches(path) == {}
    one = matched({"g3": 4})
    append_matches(path, [one])
    assert read_matches(path) == {"q1": one}
    median, p95 = search_times([one])
    assert median == pytest.approx(0.11)
    assert p95 == pytest.approx(0.11)
