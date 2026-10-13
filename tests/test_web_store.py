"""MegaLoc for all of POP: int8 rows in memory, a shortlist, exact cosines on it."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from miroir.eval.vectors import Route, RunInfo, write_vectors
from miroir.web.bundle import right_rank, write_shortlist_store
from miroir.web.engine import A3_FILE, SearchRequest, read_projection
from miroir.web.store import (
    ExactStore,
    quantize_rows,
    read_exact,
    shortlist,
    shortlist_cosines,
    write_exact,
)
from tests import web_helpers as h

INFO = RunInfo(
    option="A3",
    model="gmberton/MegaLoc",
    resolution=322,
    pooling="salad",
    gallery="g",
    created=date(2026, 10, 13),
    cost_usd=0.0,
    wall_time_s=0.0,
    route=Route.HOSTED,
)


def test_int8_rows_keep_cosines_within_a_hundredth() -> None:
    x = h.unit_rows(50, 300)
    rows, scale = quantize_rows(x)
    store = ExactStore(rows=rows, scale=scale)
    query = x[7]
    exact = store.cosines(query, np.arange(50, dtype=np.int64))
    assert np.abs(exact - x @ query).max() < 0.01
    assert int(np.argmax(exact)) == 7


def test_rows_are_quantized_in_the_order_asked() -> None:
    x = h.unit_rows(5, 16)
    order = np.asarray([4, 0, 2], dtype=np.int64)
    rows, scale = quantize_rows(x, order)
    assert np.allclose(rows * scale[:, None], x[order], atol=0.01)


def test_the_shortlist_joins_both_models_and_skips_filtered_images() -> None:
    a7 = np.asarray([0.9, 0.8, -np.inf, 0.1, 0.0], dtype=np.float32)
    reduced = np.asarray([0.0, 0.1, 0.99, 0.2, 0.7], dtype=np.float32)
    reduced[2] = -np.inf  # a filter removed image 2 from both
    assert shortlist(a7, reduced, 2).tolist() == [0, 1, 3, 4]


def test_only_shortlisted_images_get_an_exact_cosine() -> None:
    x = h.unit_rows(6, 8)
    rows, scale = quantize_rows(x)
    store = ExactStore(rows=rows, scale=scale)
    a7 = np.asarray([0.9, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reduced = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    got = shortlist_cosines(store, x[5], a7, reduced, size=1)
    assert np.isfinite(got).tolist() == [True, False, False, False, False, True]
    assert got[5] == pytest.approx(1.0, abs=0.01)


def test_a_shortlist_engine_ranks_as_the_float_engine() -> None:
    for building in (True, False):
        plain, _ = h.engine(building=building)
        short, _ = h.engine(building=building)
        rows, scale = quantize_rows(h.unit_rows(6, 8))
        short.exact = ExactStore(rows=rows, scale=scale)
        for shade in (10, 50, 130, 210):
            want = plain.search(SearchRequest(image=h.photo(shade)))
            got = short.search(SearchRequest(image=h.photo(shade)))
            assert [c.entity_id for c in got.cards] == [c.entity_id for c in want.cards]
            assert [c.image_url for c in got.cards] == [c.image_url for c in want.cards]
            assert got.confidence == pytest.approx(want.confidence, abs=0.01)


def test_a_shortlist_bundle_stores_reduced_rows_and_int8_rows(tmp_path: Path) -> None:
    x = h.unit_rows(40, 24)
    run = write_vectors(tmp_path / "run", [f"i{n:02d}" for n in range(40)], x, INFO)
    folder = tmp_path / "bundle"
    folder.mkdir()
    ids = [f"i{n:02d}" for n in reversed(range(40))]
    write_shortlist_store(folder, run, ids, dims=8)
    reduced = np.load(folder / A3_FILE)
    assert reduced.shape == (40, 8)
    projection = read_projection(folder)
    assert projection is not None
    assert np.allclose(projection.apply(x[39]), reduced[0], atol=1e-2)
    store = read_exact(folder)
    assert store is not None
    assert store.cosines(x[39], np.asarray([0], dtype=np.int64))[0] == pytest.approx(1, abs=0.01)


def test_a_bundle_without_int8_rows_has_no_store(tmp_path: Path) -> None:
    assert read_exact(tmp_path) is None
    write_exact(tmp_path, h.unit_rows(3, 4))
    assert read_exact(tmp_path) is not None


def test_a_right_answer_off_the_shortlist_is_a_miss_not_a_skipped_query() -> None:
    fused = np.asarray([0.9, -np.inf, 0.2], dtype=np.float32)
    assert right_rank(fused, [2]) == 2
    assert right_rank(fused, [1]) == 4  # past every entity: counts against hit@k
    assert right_rank(fused, []) == 0
