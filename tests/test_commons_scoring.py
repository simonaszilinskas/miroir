"""Scoring Commons photos: query runs apart from the gallery, T8c, views, fetched galleries."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from miroir.commons.model import CommonsSet, View
from miroir.config import Settings
from miroir.eval import run as run_module
from miroir.eval.run import (
    not_found_queries,
    publishable_only,
    score_by_view,
    score_test_set,
    write_fetched_gallery,
)
from miroir.eval.sets import Query, load_queries, read_gallery, write_gallery, write_set
from miroir.eval.vectors import Route, RunInfo, write_vectors
from miroir.records.build import build
from miroir.records.store import read_records, write_commons
from tests.fakes import commons_photo

DIM = 16


def _info(option: str) -> RunInfo:
    return RunInfo(
        option=option,
        model=f"model-{option}",
        resolution=512,
        pooling="head",
        gallery="g",
        created=date(2026, 10, 12),
        cost_usd=0.0,
        wall_time_s=1.0,
        route=Route.HOSTED,
    )


@pytest.fixture
def commons_sets(settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """A publishable gallery, T2 and T2b photos of its entities, and T8c photos of none.

    Every model's gallery run puts each entity on its own axis; its query run puts each photo
    near its answer, and T8c photos between axes.
    """
    build(settings)
    records = read_records(settings.records_parquet, settings.images_parquet)
    covers = [r.cover for r in records if r.publishable][:6]
    entities = [c.entity_id for c in covers]
    assert len(set(entities)) == len(entities)
    write_gallery(settings, "g", [c.image_id for c in covers])

    photos = [
        *(commons_photo(f"File:t2-{n}.jpg", CommonsSet.T2, entities[n]) for n in range(3)),
        *(commons_photo(f"File:t2b-{n}.jpg", CommonsSet.T2B, entities[n]) for n in range(3, 6)),
        *(commons_photo(f"File:t8c-{n}.jpg", CommonsSet.T8C, f"PA9{n}") for n in range(3)),
    ]
    write_commons(photos, settings.commons_parquet)
    for set_id in (CommonsSet.T2, CommonsSet.T2B, CommonsSet.T8C):
        tune = [
            Query(
                f"{set_id}-{p.image_id}",
                set_id.value,
                p.entity_id,
                p.image_id,
                () if set_id is CommonsSet.T8C else (p.entity_id,),
                (),
            )
            for p in photos
            if p.set_id is set_id
        ]
        write_set(settings, set_id.value, tune, [])

    rng = np.random.default_rng(0)
    for option in ("A", "B"):
        axes = np.eye(DIM)[: len(covers)]
        write_vectors(
            settings.vectors_path(f"{option}-gallery"),
            [c.image_id for c in covers],
            axes,
            _info(option),
        )
        rows = [axes[entities.index(p.entity_id)] if p.set_id is not CommonsSet.T8C else
                axes[0] + axes[1] for p in photos]  # fmt: skip
        noisy = np.asarray(rows) + rng.normal(scale=0.05, size=(len(photos), DIM))
        write_vectors(
            settings.vectors_path(f"{option}-queries"),
            [p.image_id for p in photos],
            noisy,
            _info(option),
        )
    monkeypatch.setattr(
        run_module, "eval_result_path", lambda r, s, o: tmp_path / r / f"{s}-{o}.json"
    )
    return settings


def test_closed_models_keep_commons_not_found_queries(commons_sets: Settings) -> None:
    queries = load_queries(commons_sets, "T8c", "tune")
    assert publishable_only(commons_sets, queries) == queries


def test_commons_photos_are_asked_with_the_query_run(commons_sets: Settings) -> None:
    scored = score_test_set(
        commons_sets,
        round_id="R1",
        step="A",
        option="A",
        vector_run="A-gallery",
        query_vector_run="A-queries",
        set_id="T2b",
        gallery="g",
        publishable_queries=True,
    )
    assert (scored.queries, scored.hit_at_1.value) == (3, 1.0)


def test_not_found_queries_hide_their_own_entity(commons_sets: Settings) -> None:
    queries = load_queries(commons_sets, "T8c", "tune")
    asked = not_found_queries(queries, "T8c")
    assert [q.hidden_entity for q in asked] == [q.entity_id for q in queries]
    with pytest.raises(ValueError, match="have a right answer"):
        not_found_queries(load_queries(commons_sets, "T2", "tune"), "T2")


def test_t2_splits_by_view(commons_sets: Settings) -> None:
    scored = score_test_set(
        commons_sets,
        round_id="R1",
        step="A",
        option="A",
        vector_run="A-gallery",
        query_vector_run="A-queries",
        set_id="T2",
        gallery="g",
    )
    path = run_module.eval_result_path("R1", "A", "A")
    assert scored.queries == 3
    (exterior,) = score_by_view(commons_sets, path, test_set="T2", gallery="g")
    assert (exterior.view, exterior.queries, exterior.hit_at_10.value) == (View.EXTERIOR, 3, 1.0)


def test_fetched_gallery_keeps_images_every_run_holds(commons_sets: Settings) -> None:
    gallery = read_gallery(commons_sets, "g")
    write_vectors(commons_sets.vectors_path("C-gallery"), gallery[1:], np.eye(DIM)[1:6], _info("C"))
    kept = write_fetched_gallery(commons_sets, "g", ["A-gallery", "C-gallery"])
    assert kept == gallery[1:]
    assert read_gallery(commons_sets, "g-fetched") == gallery[1:]
