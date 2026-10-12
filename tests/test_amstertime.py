from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from miroir.eval import amstertime
from tests.test_scoring import vector_set


def test_amstertime_directions() -> None:
    rng = np.random.default_rng(0)
    places = [f"{n:04d}" for n in range(20)]
    base = {p: rng.normal(size=8) for p in places}
    rows = {f"old/{p}.jpg": base[p] + rng.normal(scale=0.1, size=8) for p in places}
    rows |= {f"new/{p}.png": base[p] + rng.normal(scale=0.1, size=8) for p in places}
    rows["old/0001.jpg"] = rows["old/0000.jpg"]  # same side: must stay out of reach
    run = vector_set({k: list(v) for k, v in rows.items()})

    results = {d: amstertime.score_amstertime(run, d) for d in amstertime.Direction}
    assert len(results[amstertime.Direction.ALL].per_query) == 40
    assert len(results[amstertime.Direction.OLD].per_query) == 20
    old = {q.query_id: q for q in results[amstertime.Direction.OLD].per_query}
    assert old["old/0000.jpg"].rank <= 2  # its twin old/0001 is hidden, not a rival
    assert np.mean([q.ap for q in results[amstertime.Direction.NEW].per_query]) > 0.8


def test_check_ids_needs_one_old_and_one_new_per_place() -> None:
    places = [f"{n:04d}" for n in range(amstertime.PAIRS)]
    ids = [*(f"old/{p}.jpg" for p in places), *(f"new/{p}.png" for p in places)]
    amstertime.check_ids(ids)
    with pytest.raises(ValueError, match="1231 places"):
        amstertime.check_ids(ids[:-1])
    with pytest.raises(ValueError, match="1231 places"):
        amstertime.check_ids([*ids[:-1], "new/9999.png"])


def test_image_ids_skip_hidden_files(tmp_path: Path) -> None:
    for side, ext in (("old", "jpg"), ("new", "png")):
        (tmp_path / side).mkdir()
        (tmp_path / side / f"1367.{ext}").write_bytes(b"x")
        (tmp_path / side / ".DS_Store").write_bytes(b"x")  # shipped in the archive
    assert amstertime.image_ids(tmp_path) == ["new/1367.png", "old/1367.jpg"]
