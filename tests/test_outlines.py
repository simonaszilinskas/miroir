"""F4 outlines: which raw detector boxes the photo view draws (`miroir.outlines`)."""

from __future__ import annotations

import pytest

from miroir.outlines import (
    ALL_OUTLINES,
    MAX_OUTLINES,
    SITE_RULE,
    Box,
    OutlineRule,
    clipped,
    fake_outlines,
    iou,
    pick_outlines,
)

RULE = OutlineRule(min_score=0.2)


def box(x0: float, y0: float, x1: float, y1: float, score: float = 0.9) -> Box:
    return Box(x0, y0, x1, y1, score, "painting")


def pick(boxes: list[Box], rule: OutlineRule = RULE) -> tuple[Box, ...]:
    return pick_outlines(boxes, rule, width=1000, height=800)


def test_a_box_must_lie_inside_the_photo() -> None:
    with pytest.raises(ValueError, match="inside the photo"):
        Box(0.5, 0.1, 0.4, 0.2)
    assert clipped(-0.1, 0.2, 0.5, 1.3) == Box(0.0, 0.2, 0.5, 1.0)
    assert clipped(1.2, 0.0, 1.5, 0.5) is None


def test_pixels_never_give_an_empty_crop() -> None:
    assert box(0.1, 0.1, 0.5, 0.5).pixels(1000, 800) == (100, 80, 500, 400)
    assert Box(0.5, 0.5, 0.5001, 0.5001).pixels(100, 100) == (50, 50, 51, 51)


def test_unsure_and_tiny_boxes_are_dropped() -> None:
    sure = box(0.1, 0.1, 0.3, 0.4, 0.8)
    unsure = box(0.5, 0.1, 0.7, 0.4, 0.1)
    tiny = box(0.8, 0.8, 0.81, 0.81, 0.9)
    assert pick([sure, unsure, tiny]) == (sure,)


def test_the_same_object_found_twice_is_outlined_once() -> None:
    frame = box(0.1, 0.1, 0.4, 0.5, 0.9)
    again = box(0.11, 0.1, 0.41, 0.5, 0.6)
    canvas = box(0.15, 0.15, 0.35, 0.45, 0.5)  # inside the frame
    other = box(0.6, 0.1, 0.9, 0.5, 0.4)
    assert iou(frame, again) > RULE.overlap
    assert pick([canvas, other, again, frame]) == (frame, other)


def test_a_box_as_big_as_the_photo_is_never_an_outline() -> None:
    whole = box(0.0, 0.0, 1.0, 0.95, 0.99)
    painting = box(0.2, 0.2, 0.4, 0.5, 0.5)
    other = box(0.6, 0.2, 0.8, 0.5, 0.5)
    assert pick([whole, painting, other]) == (painting, other)


def test_a_photo_of_one_object_shows_no_outline() -> None:
    assert pick([box(0.1, 0.05, 0.9, 0.95)]) == ()
    # The object, surest, covers most of the photo; its stand and a detail do not count.
    vase = box(0.2, 0.05, 0.8, 0.8, 0.6)
    stand = box(0.25, 0.8, 0.75, 1.0, 0.4)
    detail = box(0.4, 0.3, 0.5, 0.4, 0.3)
    assert pick([stand, detail, vase]) == ()
    small = box(0.1, 0.1, 0.4, 0.4)
    assert pick([small]) == (small,)
    # A large box that is not the surest (a whole wall) does not hide the paintings on it.
    wall = box(0.0, 0.0, 0.7, 0.7, 0.3)
    assert pick([wall, small]) == (small, wall)


def test_outlines_are_capped_surest_first() -> None:
    boxes = [
        box(
            0.04 * (i // 2),
            0.1 + 0.4 * (i % 2),
            0.04 * (i // 2) + 0.035,
            0.3 + 0.4 * (i % 2),
            0.3 + 0.01 * i,
        )
        for i in range(50)
    ]
    shown = pick(boxes)
    assert len(shown) == ALL_OUTLINES
    assert [b.score for b in shown] == sorted((b.score for b in boxes), reverse=True)[:40]
    first = pick(boxes, OutlineRule(min_score=0.2, max_outlines=MAX_OUTLINES))
    assert first == shown[:MAX_OUTLINES]


def test_fake_outlines_depend_on_the_photo() -> None:
    a, b = fake_outlines(b"one photo"), fake_outlines(b"another photo")
    assert len(a.boxes) == 3
    assert a != b
    assert pick(list(a.boxes)) == a.boxes


def test_a_box_around_several_works_comes_after_the_works() -> None:
    wall = box(0.05, 0.05, 0.6, 0.6, 0.9)
    left = box(0.1, 0.1, 0.25, 0.3, 0.5)
    right = box(0.35, 0.1, 0.5, 0.3, 0.4)
    alone = box(0.7, 0.1, 0.9, 0.4, 0.3)
    assert pick([wall, left, right, alone], SITE_RULE) == (left, right, alone, wall)
