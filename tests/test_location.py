from __future__ import annotations

import numpy as np

from miroir.eval.location import (
    NOTHING_LEFT,
    Change,
    Places,
    Searched,
    confidences,
    entity_fields,
    hard,
    known,
    rank,
    soft,
)
from miroir.records.mappers import PlaceHint
from tests.test_geo import _record


def _searched(e7: list[list[float]], e3: list[list[float]], w: list[float]) -> Searched:
    n = len(e7)
    return Searched(
        set_id="T2",
        query_ids=tuple(f"q{i}" for i in range(n)),
        query_images=tuple(f"img{i}" for i in range(n)),
        query_entities=("A",) * n,
        right=(("A",),) * n,
        entity_ids=("A", "B", "C"),
        e7=np.asarray(e7, dtype=np.float32),
        e3=np.asarray(e3, dtype=np.float32),
        a7_weight=np.asarray(w, dtype=np.float64),
        gallery_images=3,
    )


def test_no_change_ranks_the_routed_sum() -> None:
    # A3 alone (weight 0): B first, A second.
    searched = _searched([[0.9, 0.1, 0.0]], [[0.5, 0.8, 0.0]], [0.0])
    assert rank(searched, Change()).per_query[0].rank == 2


def test_a_mask_removing_the_answer_is_a_miss_not_a_skip() -> None:
    searched = _searched([[0.9, 0.1, 0.0]], [[0.5, 0.8, 0.0]], [0.0])
    mask = np.asarray([[False, True, True]], dtype=np.bool_)
    result = rank(searched, Change(mask=mask))
    assert result.skipped_no_answer == 0
    assert result.per_query[0].rank == 4
    assert result.per_query[0].ap == 0.0


def test_hard_filter_keeps_near_and_unplaced_entities() -> None:
    d = np.asarray([[0.3, 5.0, np.nan]])
    near = hard(d, 2.0).mask
    assert near is not None
    assert near.tolist() == [[True, False, True]]
    placed_only = hard(d, 2.0, keep_unplaced=False).mask
    assert placed_only is not None
    assert placed_only.tolist() == [[True, False, False]]
    searched = _searched([[0.9, 0.1, 0.0]], [[0.5, 0.8, 0.0]], [0.0])
    assert rank(searched, hard(d, 2.0)).per_query[0].rank == 1


def test_soft_prior_lifts_the_nearby_entity() -> None:
    d = np.asarray([[0.05, 3.0, np.nan]])
    change = soft(d, 0.5, 0.5)
    assert change.bonus is not None
    assert change.bonus[0, 2] == 0.0
    assert change.bonus[0, 0] > change.bonus[0, 1]
    searched = _searched([[0.9, 0.1, 0.0]], [[0.5, 0.8, 0.0]], [0.0])
    assert rank(searched, change).per_query[0].rank == 1


def test_known_field_keeps_entities_with_the_same_value_or_none() -> None:
    searched = _searched([[0.9, 0.1, 0.0]], [[0.5, 0.8, 0.0]], [0.0])
    change = known(searched, {"A": "Gard", "B": "Aude"})
    assert change.mask is not None
    assert change.mask.tolist() == [[True, False, True]]
    unknown_answer = known(searched, {"B": "Aude"}).mask
    assert unknown_answer is not None
    assert unknown_answer.tolist() == [[True, True, True]]


def test_confidence_is_the_floor_when_nothing_is_left() -> None:
    searched = _searched([[0.9, 0.1, 0.0]], [[0.5, 0.8, 0.0]], [0.5])
    found = confidences(searched, Change())
    assert found[0].score == np.float32(0.5 * 0.9 + 0.5 * 0.5)
    empty = confidences(searched, Change(mask=np.zeros((1, 3), dtype=bool)))
    assert empty[0].score == NOTHING_LEFT


def test_places_distances_with_noise_move_by_the_noise() -> None:
    searched = _searched([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [0.0])
    places = Places(
        lat=np.asarray([48.85, 48.85, np.nan]),
        lon=np.asarray([2.35, 2.40, np.nan]),
        query={"img0": (48.85, 2.35)},
    )
    exact = places.distances(searched)
    assert exact[0, 0] == 0.0
    assert np.isnan(exact[0, 2])
    noisy = places.distances(searched, noise_km=0.3)
    assert abs(noisy[0, 0] - 0.3) < 1e-6


def test_entity_fields_take_the_first_value_of_an_entity() -> None:
    first = _record("AP1", "PA1")
    records = [first, _record("AP2", "PA1"), _record("J1", "J1")]
    hints = {"AP2": PlaceHint("AP2", "31555", None), "J1": PlaceHint("J1", None, "M0563")}
    fields = entity_fields(records, hints)
    assert fields["commune"] == {"PA1": "31555"}
    assert fields["museum"] == {"J1": "M0563"}
    assert fields["departement"] == {}
