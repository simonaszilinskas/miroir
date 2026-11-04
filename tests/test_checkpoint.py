from __future__ import annotations

from pathlib import Path

import numpy as np

from miroir.jobs.checkpoint import Checkpoint


def test_checkpoint_reopens_with_what_it_held(tmp_path: Path) -> None:
    first = Checkpoint(tmp_path / "run")
    first.append(["a", "b"], np.ones((2, 3), dtype=np.float16))
    first.append(["c"], np.zeros((1, 3), dtype=np.float16))
    again = Checkpoint(tmp_path / "run")
    assert again.ids == ("a", "b", "c")
    assert again.vectors().shape == (3, 3)


def test_a_half_written_chunk_is_cut_off(tmp_path: Path) -> None:
    cp = Checkpoint(tmp_path / "run")
    cp.append(["a"], np.ones((1, 3), dtype=np.float16))
    with (tmp_path / "run" / "vectors.f16").open("ab") as out:
        out.write(b"\x00" * 4)  # rows written, crash before their ids
    again = Checkpoint(tmp_path / "run")
    assert again.ids == ("a",)
    again.append(["b"], np.zeros((1, 3), dtype=np.float16))
    np.testing.assert_array_equal(Checkpoint(tmp_path / "run").vectors()[1], [0, 0, 0])


def test_a_half_written_last_id_is_dropped_with_its_row(tmp_path: Path) -> None:
    cp = Checkpoint(tmp_path / "run")
    cp.append(["a"], np.ones((1, 3), dtype=np.float16))
    with (tmp_path / "run" / "vectors.f16").open("ab") as out:
        out.write(np.zeros(3, dtype=np.float16).tobytes())  # a whole row ...
    with (tmp_path / "run" / "ids.txt").open("ab") as out:
        out.write(b"bc")  # ... and its id cut short: "bcd" was being written
    again = Checkpoint(tmp_path / "run")
    assert again.ids == ("a",)
    again.append(["e"], np.full((1, 3), 2, dtype=np.float16))
    final = Checkpoint(tmp_path / "run")
    assert final.ids == ("a", "e")
    np.testing.assert_array_equal(final.vectors()[1], [2, 2, 2])


def test_ids_past_the_last_whole_row_are_dropped_not_refused(tmp_path: Path) -> None:
    cp = Checkpoint(tmp_path / "run")
    cp.append(["a", "b"], np.ones((2, 3), dtype=np.float16))
    vectors = tmp_path / "run" / "vectors.f16"
    vectors.write_bytes(vectors.read_bytes()[:8])  # the disk lost most of the second row
    again = Checkpoint(tmp_path / "run")
    assert again.ids == ("a",)
    assert Checkpoint(tmp_path / "run").ids == ("a",)


def test_only_keeps_the_ids_asked_for_once_each(tmp_path: Path) -> None:
    cp = Checkpoint(tmp_path / "run")
    cp.append(["a", "x", "b"], np.arange(9, dtype=np.float16).reshape(3, 3))
    cp.append(["a"], np.zeros((1, 3), dtype=np.float16))
    ids, vectors = cp.only(["b", "a"])
    assert ids == ("a", "b")
    np.testing.assert_array_equal(vectors, [[0, 1, 2], [6, 7, 8]])
