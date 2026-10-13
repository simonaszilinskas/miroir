"""Set the margin that makes an object search "sure" (`miroir.web.sure`), on tune halves.

The site's own ranking on its index (`miroir.eval.site_rerank.search_sets`, cached) gives each
tune query its top entities and their confidences. The margin chosen is the smallest on a
grid that keeps both promises:

- right: at least `RIGHT_TARGET` of sure object answers are the right record, on T3, T5
  and T2b together and on each of them;
- honest: at most `WRONG_TARGET` of T8 object searches (their whole entity removed from the
  index, so any answer is wrong) still look sure.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from miroir.config import WEB_INDEX, Settings, eval_result_path
from miroir.eval.site_rerank import Searched
from miroir.web.bundle import WRONG_TARGET
from miroir.web.routing import PhotoType
from miroir.web.sure import SURE_FILE, SureRule, margin_of, write_sure

logger = logging.getLogger(__name__)

RIGHT_TARGET = 0.70
RANKED = ("T3", "T5", "T2b")
GRID = tuple(round(0.005 * i, 3) for i in range(81))  # 0 to 0.4


@dataclass(frozen=True, slots=True)
class SureRow:
    margin: float
    sure: dict[str, float]
    """Per set: share of object searches answered "sure"."""
    right: dict[str, float]
    """Per set: share of sure answers that are the right record."""
    t8_sure: float
    all_right: float
    all_sure: float


def _object_rows(s: Searched, *, answered: bool) -> list[tuple[float, bool]]:
    return [
        (margin_of(h.confidences), h.right_rank == 1)
        for h in s.hits
        if h.photo_type is PhotoType.OBJECT and (h.right_rank > 0 or not answered)
    ]


def sure_rows(searched: Sequence[Searched]) -> list[SureRow]:
    sets = {s.set_id: s for s in searched}
    ranked = {k: _object_rows(sets[k], answered=True) for k in RANKED if k in sets}
    t8 = np.asarray([m for m, _ in _object_rows(sets["T8"], answered=False)])
    rows: list[SureRow] = []
    for m in GRID:
        sure: dict[str, float] = {}
        right: dict[str, float] = {}
        hits: list[bool] = []
        count = 0
        for k, values in ranked.items():
            got = [ok for margin, ok in values if margin >= m]
            sure[k] = len(got) / len(values)
            right[k] = float(np.mean(got)) if got else 0.0
            hits += got
            count += len(values)
        rows.append(
            SureRow(
                margin=m,
                sure=sure,
                right=right,
                t8_sure=float((t8 >= m).mean()),
                all_right=float(np.mean(hits)) if hits else 0.0,
                all_sure=len(hits) / count,
            )
        )
    return rows


def choose(rows: Sequence[SureRow]) -> SureRow:
    for row in rows:
        if (
            row.t8_sure <= WRONG_TARGET
            and row.all_right >= RIGHT_TARGET
            and all(v >= RIGHT_TARGET for k, v in row.right.items() if row.sure[k] > 0)
        ):
            return row
    msg = "no margin on the grid keeps both promises"
    raise ValueError(msg)


def write_sure_rule(
    settings: Settings, searched: Sequence[Searched], *, round_id: str = "R3"
) -> SureRow:
    """Write `sure.json` into the bundle, keep a copy for `build-index`, record the numbers."""
    rows = sure_rows(searched)
    chosen = choose(rows)
    before = next(r for r in rows if r.margin == 0.0)
    date = datetime.now(UTC).date().isoformat()
    note = (
        f"Smallest margin keeping >= {RIGHT_TARGET:.0%} of sure object answers right on T3, T5, "
        f"T2b tune and <= {WRONG_TARGET:.0%} of T8 object searches sure (index {WEB_INDEX}, "
        f"{date}): {chosen.all_sure:.0%} of object searches sure, {chosen.all_right:.0%} of "
        f"them right; T8 {chosen.t8_sure:.0%}"
    )
    rule = SureRule(margin=chosen.margin, note=note)
    bundle = settings.web_index_dir(WEB_INDEX)
    write_sure(bundle / SURE_FILE, rule)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / SURE_FILE, settings.index_dir / f"sure-{WEB_INDEX}.json")
    detail = eval_result_path(round_id, "G", "sure-margin").parent / "G-detail" / "sure-margin.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(
        json.dumps(
            {
                "index": WEB_INDEX,
                "chosen": _row(chosen),
                "margin_0": _row(before),
                "grid": [_row(r) for r in rows],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("%s", note)
    return chosen


def _row(row: SureRow) -> dict[str, object]:
    return {
        "margin": row.margin,
        "sure": row.sure,
        "right": row.right,
        "t8_sure": row.t8_sure,
        "all_sure": row.all_sure,
        "all_right": row.all_right,
    }
