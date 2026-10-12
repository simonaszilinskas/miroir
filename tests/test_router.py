from __future__ import annotations

import numpy as np
import pytest

from miroir.config import SEED
from miroir.eval.fusion import Method, fuse, fuse_per_query, min_max
from miroir.eval.router import (
    PhotoType,
    Route,
    RoutedSet,
    fit_logistic,
    photo_type,
    pick_route,
    rank_routed,
    routed,
)
from miroir.eval.scoring import EntityScores, SearchQuery, rank_entities
from miroir.records.model import Collection, Record


def record(collection: Collection, entity_id: str) -> Record:
    return Record(
        ref="R1",
        collection=collection,
        entity_id=entity_id,
        title="",
        description="",
        denomination=(),
        domain=(),
        authors=(),
        period=(),
        region=None,
        departement=None,
        commune=None,
        publishable=True,
        pop_url="",
        linked_refs=(),
        images=(),
    )


def test_photo_type_reads_the_base_and_the_linked_id() -> None:
    assert photo_type(record(Collection.JOCONDE, "000PE001")) is PhotoType.OBJECT
    assert photo_type(record(Collection.ROSE_VALLAND, "MNR00662")) is PhotoType.OBJECT
    assert photo_type(record(Collection.MEMOIRE, "PM22000637")) is PhotoType.OBJECT
    assert photo_type(record(Collection.MEMOIRE, "PA00087461")) is PhotoType.BUILDING
    assert photo_type(record(Collection.MEMOIRE, "APMH0226399")) is None


def test_logistic_regression_separates_two_clouds() -> None:
    rng = np.random.default_rng(SEED)
    x = np.vstack([rng.normal(-1, 1, (200, 3)), rng.normal(1, 1, (200, 3))]).astype(np.float32)
    y = np.r_[np.zeros(200), np.ones(200)]
    coef, bias = fit_logistic(x, y, np.ones(400), penalty=1e-3)
    guessed = (x @ coef + bias) > 0
    assert (guessed == (y > 0.5)).mean() > 0.9
    assert (coef > 0).all()


def scores(*rows: list[float]) -> EntityScores:
    values = np.asarray(rows, dtype=np.float32)
    return EntityScores(
        query_ids=tuple(f"q{i}" for i in range(len(rows))),
        entity_ids=tuple("ABC"[: values.shape[1]]),
        values=values,
        gallery_images=values.shape[1],
    )


def test_constant_per_query_weights_match_g1() -> None:
    a, b = scores([1.0, 0.9, 0.0], [0.2, 0.4, 0.3]), scores([0.0, 1.0, 0.5], [0.9, 0.1, 0.5])
    per_query = fuse_per_query([a, b], np.full((2, 2), 0.5, dtype=np.float32))
    plain = fuse([a, b], (0.5, 0.5), Method.WEIGHTED_SUM)
    np.testing.assert_allclose(per_query.values, plain.values)


# Model a knows q0's answer (A), model b knows q1's answer (C). q0 is a building, q1 an object.
A_SCORES = scores([1.0, 0.5, 0.0], [1.0, 0.5, 0.0])
B_SCORES = scores([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
QUERIES = (SearchQuery("q0", "i0", ("A",)), SearchQuery("q1", "i1", ("C",)))


def routed_set(p_building: list[float]) -> RoutedSet:
    parts = (A_SCORES, B_SCORES)
    return RoutedSet(
        QUERIES, parts, tuple(min_max(p.values) for p in parts), np.asarray(p_building)
    )


def test_route_sends_each_photo_to_its_model() -> None:
    rs = routed_set([0.9, 0.1])
    ranks = [q.rank for q in rank_entities(routed(rs, Route(1.0, 0.0)), QUERIES).per_query]
    assert ranks == [1, 1]
    hard = Route(0.8, 0.2, hard=True)
    assert hard.first_weights(np.asarray([0.6, 0.4])).tolist() == [0.8, 0.2]


def test_fast_ranking_matches_the_plain_one() -> None:
    rs = routed_set([0.7, 0.2])
    route = Route(0.6, 0.3)
    assert rank_routed(rs, route) == rank_entities(routed(rs, route), QUERIES)


def test_grid_picks_split_weights_when_one_weight_cannot_serve_both() -> None:
    route, (hit10, _) = pick_route([routed_set([1.0, 0.0])], hard=False)
    assert hit10 == 1.0
    assert route.building > route.object
    tied, _ = pick_route([routed_set([1.0, 0.0])], hard=False, tied=True)
    assert tied.building == tied.object


def test_per_query_weights_refuse_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match="per-query weights"):
        fuse_per_query([A_SCORES, B_SCORES], np.ones((3, 2), dtype=np.float32))
