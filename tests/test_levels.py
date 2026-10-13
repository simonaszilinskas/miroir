"""Card levels: the rule (`miroir.web.levels`) and how its thresholds are set (`eval.levels`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from miroir.eval.levels import Ranked, choose_levels, choose_route, not_found_shares
from miroir.web.levels import (
    LEVEL_LABELS,
    Level,
    Levels,
    RouteLevels,
    gap_of,
    read_levels,
    write_levels,
)

RULE = RouteLevels(
    "object", very_close_score=0.5, very_close_gap=0.1, close_score=0.3, close_gap=0.05
)


def test_the_first_card_reads_its_score_and_its_gap() -> None:
    assert RULE.first_level([0.9, 0.7]) is Level.VERY_CLOSE
    assert RULE.first_level([0.9, 0.87]) is Level.LIKENESS  # high, but a near tie
    assert RULE.first_level([0.9, 0.83]) is Level.CLOSE
    assert RULE.first_level([0.4, 0.1]) is Level.CLOSE  # stands out, but a low score
    assert RULE.first_level([0.2, 0.0]) is Level.LIKENESS
    assert RULE.first_level([]) is Level.LIKENESS


def test_cards_after_the_first_are_likenesses() -> None:
    assert RULE.card_levels(3, [0.9, 0.7, 0.6]) == [
        Level.VERY_CLOSE,
        Level.LIKENESS,
        Level.LIKENESS,
    ]
    assert RULE.card_levels(0, [0.9]) == []


def test_a_missing_pair_is_never_reached() -> None:
    words = RouteLevels("words", None, None, 0.0, 0.0)
    assert words.first_level([1.0, 0.0]) is Level.CLOSE


def test_labels_are_short_french_words() -> None:
    assert [LEVEL_LABELS[lv] for lv in Level] == ["Très proche", "Proche", "Ressemblance"]


def test_gap_is_to_the_best_other_card() -> None:
    assert gap_of([0.9, 0.7, 0.85]) == pytest.approx(0.05)
    assert gap_of([0.9]) == pytest.approx(0.9)


def test_levels_read_back(tmp_path: Path) -> None:
    levels = Levels(routes=(RULE,), source="test")
    write_levels(tmp_path / "levels.json", levels)
    assert read_levels(tmp_path / "levels.json") == levels
    assert levels.for_route("object") == RULE
    assert levels.for_route("words") is None
    assert read_levels(tmp_path / "missing.json") is None


def ranked(set_id: str, first: float, gap: float, right: bool, i: int) -> Ranked:
    return Ranked(
        set_id, f"{set_id}-{i}", "object", (first, first - gap, 0.1), (right, False, False)
    )


def searches() -> list[Ranked]:
    """Right answers stand out by 0.2; near ties (0.03) are right half the time; flat ones
    (0.005) never; not-found searches stand out by at most 0.02."""
    rows = [ranked("T5", 0.9, 0.2, True, i) for i in range(18)]
    rows += [ranked("T5", 0.9, 0.2, False, 100 + i) for i in range(2)]
    rows += [ranked("T5", 0.8, 0.03, i % 2 == 0, 200 + i) for i in range(30)]
    rows += [ranked("T5", 0.7, 0.005, False, 300 + i) for i in range(30)]
    rows += [ranked("T8", 0.8, 0.005 * (i % 5), False, 400 + i) for i in range(30)]
    return rows


def test_thresholds_keep_their_promises() -> None:
    rows = searches()
    rule = choose_route("object", rows)
    asked = [r for r in rows if r.set_id == "T5"]
    vc = [r.right[0] for r in asked if rule.first_level(r.scores) is Level.VERY_CLOSE]
    close = [r.right[0] for r in asked if rule.first_level(r.scores) is Level.CLOSE]
    assert len(vc) == 20
    assert sum(vc) / len(vc) >= 0.8
    assert close
    assert sum(close) / len(close) >= 0.4
    lost = [r for r in rows if r.set_id == "T8"]
    assert all(rule.first_level(r.scores) is not Level.VERY_CLOSE for r in lost)
    assert sum(rule.first_level(r.scores) is not Level.LIKENESS for r in lost) <= 0.2 * len(lost)


def test_the_report_counts_each_level_and_not_found_searches() -> None:
    rows = searches()
    (report,) = choose_levels(rows)
    assert report.searches == 80
    assert report.levels[Level.VERY_CLOSE.value].searches == pytest.approx(0.25)
    assert report.levels[Level.VERY_CLOSE.value].right == pytest.approx(0.9)
    assert not_found_shares(rows, [report]) == {"T8": 0.0}


def test_no_pair_keeping_the_promise_means_no_very_close() -> None:
    rows = [ranked("T7", 0.9, 0.2, i % 2 == 0, i) for i in range(40)]
    rule = choose_route("words", rows)
    assert rule.very_close_score is None
    assert rule.close_score is not None  # half right: enough for "Proche"
