from __future__ import annotations

import numpy as np
import pytest

from miroir.eval.expansion import expand_queries
from miroir.eval.fusion import RRF_K, Method, fuse, min_max, ranks, split_weight
from miroir.eval.scoring import (
    EntityScores,
    Gallery,
    SearchQuery,
    entity_scores,
    rank_entities,
)
from tests.test_scoring import ENTITY, GALLERY, GALLERY_RUN, vector_set

INF = float("inf")


def scores(*rows: list[float]) -> EntityScores:
    values = np.asarray(rows, dtype=np.float32)
    return EntityScores(
        query_ids=tuple(f"q{i}" for i in range(len(rows))),
        entity_ids=tuple("ABCD"[: values.shape[1]]),
        values=values,
        gallery_images=values.shape[1],
    )


def test_entity_scores_then_ranking_matches_the_scorer() -> None:
    query = SearchQuery("q1", "q", ("X",))
    found = entity_scores(GALLERY_RUN, GALLERY, [query])
    assert found.entity_ids == ("X", "Y", "Z")
    assert found.values[0] == pytest.approx([0.8, 0.6, 1.0], abs=1e-3)
    assert rank_entities(found, [query]).per_query[0].rank == 2


def test_min_max_keeps_hidden_entities_at_the_bottom() -> None:
    assert min_max(scores([0.2, 0.6, -INF, 1.0]).values)[0] == pytest.approx([0, 0.5, 0, 1])


def test_ranks_break_ties_by_entity_order() -> None:
    assert ranks(scores([0.5, 0.9, 0.5]).values)[0].tolist() == [2, 1, 3]


def test_weighted_sum_lets_a_second_model_overturn_a_close_call() -> None:
    a = scores([1.0, 0.95, 0.0])  # A just ahead of B
    b = scores([0.0, 1.0, 0.5])  # B far ahead
    fused = fuse([a, b], (0.5, 0.5), Method.WEIGHTED_SUM)
    assert int(np.argmax(fused.values[0])) == 1


def test_reciprocal_rank_fusion_uses_ranks_only() -> None:
    a = scores([0.9, 0.1, 0.0])
    b = scores([0.0, 100.0, 50.0])
    fused = fuse([a, b], (0.5, 0.5), Method.RECIPROCAL_RANK).values[0]
    expected_a = 0.5 / (RRF_K + 1) + 0.5 / (RRF_K + 3)
    assert fused[0] == pytest.approx(expected_a)


def test_fusion_keeps_hidden_entities_hidden() -> None:
    fused = fuse([scores([-INF, 1.0]), scores([-INF, 0.5])], (0.5, 0.5), Method.WEIGHTED_SUM)
    assert fused.values[0, 0] == -INF


def test_fusion_refuses_models_on_different_entities() -> None:
    other = EntityScores(("q0",), ("A", "Z"), np.zeros((1, 2), dtype=np.float32), 2)
    with pytest.raises(ValueError, match="different queries or galleries"):
        fuse([scores([0.1, 0.2]), other], (0.5, 0.5), Method.WEIGHTED_SUM)


def test_split_weight_shares_the_rest() -> None:
    assert split_weight(0.6, 3) == pytest.approx((0.6, 0.2, 0.2))
    with pytest.raises(ValueError, match="first weight"):
        split_weight(1.0, 2)


def test_expansion_moves_the_query_towards_its_hits() -> None:
    query = SearchQuery("q1", "q", ("X",))
    expanded = expand_queries(GALLERY_RUN, GALLERY, [query], top=1)
    # Top hit of q = [0, 1] is g3 = [0, 1]: the average stays on it.
    assert expanded.vectors[0] == pytest.approx([0, 1], abs=1e-3)
    expanded = expand_queries(GALLERY_RUN, GALLERY, [query], top=2)
    # Adds g4 = [0.6, 0.8]: the query turns towards entity X.
    direction = np.asarray([0.6, 2.8]) / np.linalg.norm([0.6, 2.8])
    assert expanded.vectors[0] == pytest.approx(direction, abs=1e-3)


def test_expansion_never_uses_a_hidden_image() -> None:
    query = SearchQuery("q1", "q", ("X",), excluded_image_ids=("g3",))
    expanded = expand_queries(GALLERY_RUN, GALLERY, [query], top=1)
    direction = np.asarray([0.6, 1.8]) / np.linalg.norm([0.6, 1.8])  # g4, not g3
    assert expanded.vectors[0] == pytest.approx(direction, abs=1e-3)


def test_expansion_alpha_weights_hits_by_similarity() -> None:
    run = vector_set({"a": [1, 0], "b": [0, 1], "q": [1, 1]})
    gallery = Gallery.from_vectors(run, ENTITY | {"a": "A", "b": "B"}, ["a", "b"])
    query = SearchQuery("q1", "q", ("A",))
    plain = expand_queries(run, gallery, [query], top=2, alpha=0)
    weighted = expand_queries(run, gallery, [query], top=2, alpha=3)
    assert plain.vectors[0] == pytest.approx(weighted.vectors[0], abs=1e-3)  # equal sims
