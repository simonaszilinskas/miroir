from __future__ import annotations

import numpy as np

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
