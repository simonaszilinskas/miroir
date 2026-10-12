from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from miroir.eval.notfound import (
    Confidence,
    NotFoundScore,
    confidences,
    match_rate_at,
    not_found_score,
    wrong_confident_rate,
)
from miroir.eval.results import (
    SUMMARY_END,
    SUMMARY_START,
    read_result,
    rebuild_summary,
    write_result,
)
from miroir.eval.scoring import EntityScores, Interval
from tests.test_scoring import INFO

FOUND = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
MISSING = np.asarray([0.85, 0.65, 0.45, 0.25, 0.05])


def test_wrong_confident_at_half_matched() -> None:
    # Half the in-index queries match at cut-off 0.5; 0.85 and 0.65 of T8 pass it.
    assert wrong_confident_rate(FOUND, MISSING, 0.5) == pytest.approx(2 / 5)


def test_match_rate_at_a_wrong_target() -> None:
    # 0 of 5 T8 queries allowed: the cut-off sits above 0.85, letting only 0.9 through.
    assert match_rate_at(FOUND, MISSING, 0.05) == pytest.approx(1 / 10)
    # 1 of 5 allowed (20%): above 0.65, so 0.9, 0.8, 0.7 match.
    assert match_rate_at(FOUND, MISSING, 0.2) == pytest.approx(3 / 10)


def test_confidence_scores_the_entity_ranked_first() -> None:
    def part(*values: float) -> EntityScores:
        return EntityScores(("q",), ("A", "B"), np.asarray([values], dtype=np.float32), 2)

    a, b = part(0.9, 0.8), part(0.1, 0.5)
    ranking = part(0.0, 1.0)  # fusion put B first
    (only,) = confidences(ranking, [a, b], [0.5, 0.5])
    assert only.score == pytest.approx(0.5 * 0.8 + 0.5 * 0.5)


def _score() -> NotFoundScore:
    found = [Confidence(f"T1-{i}", float(v)) for i, v in enumerate(FOUND)]
    missing = [Confidence(f"T8-{i}", float(v)) for i, v in enumerate(MISSING)]
    return not_found_score(
        found, missing, test_set="T8", half="tune", gallery="fixture", in_index_sets=("T1",)
    )


def test_not_found_score_has_intervals_around_its_values() -> None:
    scored = _score()
    assert scored.wrong_confident.value == pytest.approx(0.4)
    assert scored.wrong_confident.low <= 0.4 <= scored.wrong_confident.high


def test_not_found_score_refuses_a_value_that_does_not_match() -> None:
    scored = _score()
    with pytest.raises(ValueError, match="from the confidences"):
        replace(scored, wrong_confident=Interval(0.1, 0.0, 0.2))


def test_not_found_round_trip_and_summary(tmp_path: Path) -> None:
    path = tmp_path / "results" / "R2" / "A-test.json"
    write_result(
        path,
        round_id="R2",
        step="A",
        option="test",
        vector_run="run1",
        query_vector_run="run1",
        vectors=INFO,
        scores=[],
        not_found=[_score()],
        today=date(2026, 10, 12),
    )
    (only,) = read_result(path).not_found
    assert only.wrong_confident.value == pytest.approx(0.4)
    md = tmp_path / "results.md"
    md.write_text(f"{SUMMARY_START}\n{SUMMARY_END}\n")
    rebuild_summary(tmp_path / "results", md)
    assert "| R2 | T8 | tune | fixture | A-test | T1 | 10 | 5 | 40.0 [" in md.read_text()
