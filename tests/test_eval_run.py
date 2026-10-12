from __future__ import annotations

import pytest

from miroir.eval.run import search_queries
from miroir.eval.sets import Query


def _query(expected: tuple[str, ...]) -> Query:
    return Query(
        query_id="T1-1",
        set_id="T1",
        entity_id="PA1",
        query_image_id="img1",
        expected_entity_ids=expected,
        exclude_image_ids=("img1", "img2"),
    )


def test_test_set_queries_become_search_queries() -> None:
    (query,) = search_queries([_query(("PA1",))])
    assert query.image_id == "img1"
    assert query.entity_ids == ("PA1",)
    assert query.excluded_image_ids == ("img1", "img2")


def test_not_found_queries_are_refused() -> None:
    with pytest.raises(ValueError, match="T8"):
        search_queries([_query(())])
