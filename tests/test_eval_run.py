from __future__ import annotations

import pytest

from miroir.config import Settings
from miroir.eval.run import publishable_only, search_queries
from miroir.eval.sets import Query
from miroir.records.build import build
from miroir.records.store import read_records


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


def test_publishable_only_drops_queries_a_closed_model_may_not_see(settings: Settings) -> None:
    build(settings)
    records = read_records(settings.records_parquet, settings.images_parquet)
    public = next(r for r in records if r.publishable).cover.image_id
    private = next(r for r in records if not r.publishable).cover.image_id
    queries = [
        Query(f"T5-{n}", "T5", "E", image_id, ("E",), (image_id,))
        for n, image_id in enumerate((public, private))
    ]
    assert [q.query_image_id for q in publishable_only(settings, queries)] == [public]
