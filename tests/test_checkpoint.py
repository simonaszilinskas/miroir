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
