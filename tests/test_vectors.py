from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from miroir.eval.vectors import (
    IDS_FILE,
    META_FILE,
    VECTORS_FILE,
    Route,
    RunInfo,
    VectorFormatError,
    read_vectors,
    write_vectors,
)

INFO = RunInfo(
    option="A1",
    model="facebook/dinov2-large",
    resolution=224,
    pooling="cls",
    gallery="fixture",
    created=date(2026, 10, 12),
    cost_usd=0.0,
    wall_time_s=1.5,
    route=Route.MAC,
)


def test_round_trip_normalises_and_stores_float16(tmp_path: Path) -> None:
    raw = np.array([[3.0, 4.0], [0.0, 2.0], [1.0, 1.0]])
    write_vectors(tmp_path / "run", ["a", "b", "c"], raw, INFO)
    loaded = read_vectors(tmp_path / "run")
    assert loaded.ids == ("a", "b", "c")
    assert loaded.vectors.dtype == np.float16
    np.testing.assert_allclose(loaded.vectors[0], [0.6, 0.8], atol=1e-3)
    assert loaded.info == INFO
    assert loaded.dim == 2
    assert loaded.index()["c"] == 2


def test_meta_json_is_readable_by_hand(tmp_path: Path) -> None:
    write_vectors(tmp_path / "run", ["a"], np.ones((1, 4)), INFO)
    meta = json.loads((tmp_path / "run" / META_FILE).read_text())
    assert meta["format_version"] == 1
    assert meta["dtype"] == "float16"
    assert (meta["dim"], meta["count"]) == (4, 1)
    assert meta["info"]["route"] == "mac"
    assert meta["info"]["created"] == "2026-10-12"


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    write_vectors(tmp_path / "run", ["a"], np.ones((1, 2)), INFO)
    with pytest.raises(FileExistsError):
        write_vectors(tmp_path / "run", ["a"], np.ones((1, 2)), INFO)


@pytest.mark.parametrize(
    ("ids", "vectors", "match"),
    [
        (["a", "a"], np.ones((2, 2)), "duplicate"),
        (["a", "b c"], np.ones((2, 2)), "whitespace"),
        (["a"], np.ones((2, 2)), "does not match"),
        (["a"], np.zeros((1, 2)), "zero vector"),
        (["a"], np.array([[np.nan, 1.0]]), "NaN"),
    ],
)
def test_write_rejects_bad_input(
    tmp_path: Path, ids: list[str], vectors: np.ndarray, match: str
) -> None:
    with pytest.raises(VectorFormatError, match=match):
        write_vectors(tmp_path / "run", ids, vectors, INFO)


def _corrupt(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    write_vectors(run, ["a", "b"], np.eye(2), INFO)
    return run


def test_read_rejects_unnormalised_rows(tmp_path: Path) -> None:
    run = _corrupt(tmp_path)
    np.save(run / VECTORS_FILE, np.array([[1, 0], [0, 2]], dtype=np.float16))
    with pytest.raises(VectorFormatError, match="norm"):
        read_vectors(run)


def test_read_rejects_float32(tmp_path: Path) -> None:
    run = _corrupt(tmp_path)
    np.save(run / VECTORS_FILE, np.eye(2, dtype=np.float32))
    with pytest.raises(VectorFormatError, match="float32"):
        read_vectors(run)


def test_read_rejects_id_count_mismatch(tmp_path: Path) -> None:
    run = _corrupt(tmp_path)
    (run / IDS_FILE).write_text("a\n", encoding="utf-8")
    with pytest.raises(VectorFormatError, match="1 ids"):
        read_vectors(run)


def test_read_rejects_unknown_meta_fields(tmp_path: Path) -> None:
    run = _corrupt(tmp_path)
    meta = json.loads((run / META_FILE).read_text())
    meta["info"]["colour"] = "blue"
    (run / META_FILE).write_text(json.dumps(meta))
    with pytest.raises(VectorFormatError, match="colour"):
        read_vectors(run)


def test_unfinished_run_has_no_meta(tmp_path: Path) -> None:
    with pytest.raises(VectorFormatError, match="unfinished"):
        read_vectors(tmp_path)


def test_run_info_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="negative"):
        RunInfo(
            option="A9",
            model="google/gemini-embedding-2",
            resolution=1024,
            pooling="api",
            gallery="screening-20k",
            created=date(2026, 10, 12),
            cost_usd=-1.0,
            wall_time_s=0.0,
            route=Route.OPENROUTER,
        )
