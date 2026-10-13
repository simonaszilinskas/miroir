"""Step C scoring (tapped outline, any box, box recall) and E1 in the site (`mix`)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from miroir.eval.detect import TOP, Detections, RoomBoxes, T3Item, score_outlines
from miroir.eval.site_rerank import MIX_INLIERS, rerank_hit
from miroir.eval.wire import SiteMatches
from miroir.outlines import Box, OutlineRule
from miroir.web.bundle import SiteHit
from miroir.web.routing import PhotoType

RULE = OutlineRule(min_score=0.2)


def detections(boxes: tuple[Box, ...], ranked: tuple[tuple[str, ...], ...]) -> Detections:
    room = RoomBoxes("room", 2000, 1500, boxes, 0.05, ranked)
    return Detections(
        label="C1-test",
        model="m",
        index="full",
        created=datetime(2026, 10, 13, tzinfo=UTC),
        load_s=1.0,
        cost_usd=0.01,
        gallery_images=10,
        gallery_entities=5,
        rooms=(room,),
    )


def test_the_tapped_outline_is_the_one_covering_the_artwork() -> None:
    left = Box(0.1, 0.1, 0.3, 0.4, 0.9)
    right = Box(0.6, 0.1, 0.8, 0.4, 0.8)
    found = detections((left, right), (("B", "A"), ("A", "B")))
    items = [T3Item("q1", "A", ("room",)), T3Item("q2", "B", ("room",))]
    located = {("room", "A"): Box(0.11, 0.1, 0.3, 0.42), ("room", "B"): Box(0.6, 0.1, 0.8, 0.4)}
    score = score_outlines(found, items, located, RULE)
    # Each hangs under the outline that ranks it second; the other outline ranks it first.
    assert [q.rank for q in score.tapped.per_query] == [2, 2]
    assert [q.rank for q in score.any_box.per_query] == [1, 1]
    assert score.box_recall == 1.0
    assert score.located == 2


def test_an_artwork_with_no_outline_over_it_is_a_miss() -> None:
    found = detections((Box(0.1, 0.1, 0.3, 0.4, 0.9),), (("A",),))
    items = [T3Item("q1", "A", ("room",))]
    located = {("room", "A"): Box(0.6, 0.5, 0.9, 0.9)}
    score = score_outlines(found, items, located, RULE)
    assert score.tapped.per_query[0].rank == TOP + 1
    assert score.any_box.per_query[0].rank == 1
    assert score.box_recall == 0.0


def test_an_artwork_never_located_falls_back_to_any_box() -> None:
    found = detections((Box(0.1, 0.1, 0.3, 0.4, 0.9),), (("X", "A"),))
    score = score_outlines(found, [T3Item("q1", "A", ("room",))], {}, RULE)
    assert score.tapped.per_query[0].rank == 2
    assert score.located == 0


def hit(kind: PhotoType, right_rank: int) -> SiteHit:
    return SiteHit(
        bucket=kind.value,
        photo_type=kind,
        entities=("e1", "e2", "e3"),
        confidences=(0.80, 0.79, 0.70),
        best_images=("i1", "i2", "i3"),
        right_rank=right_rank,
    )


def test_mix_lifts_the_entity_with_many_inliers() -> None:
    matched = SiteMatches("q", ("i1", "i2", "i3"), (3, 2, 150), 0.9)
    rank, confidence = rerank_hit(hit(PhotoType.OBJECT, 3), matched)
    assert rank == 1
    assert confidence == pytest.approx(0.70 + 150 / MIX_INLIERS)
    # The right entity was first and drops behind the one that matched.
    assert rerank_hit(hit(PhotoType.OBJECT, 1), matched)[0] == 2
    # Past the shortlist, or not in the index: unchanged.
    assert rerank_hit(hit(PhotoType.OBJECT, 40), matched)[0] == 40
    assert rerank_hit(hit(PhotoType.OBJECT, 0), matched)[0] == 0


def test_buildings_are_never_re_ranked() -> None:
    matched = SiteMatches("q", ("i1", "i2", "i3"), (0, 0, 500), 0.9)
    assert rerank_hit(hit(PhotoType.BUILDING, 3), matched) == (3, 0.80)
    assert rerank_hit(hit(PhotoType.OBJECT, 3), None) == (3, 0.80)


def test_matches_against_another_shortlist_are_refused() -> None:
    matched = SiteMatches("q", ("i1", "i9", "i3"), (0, 0, 0), 0.9)
    with pytest.raises(ValueError, match="another shortlist"):
        rerank_hit(hit(PhotoType.OBJECT, 3), matched)
