"""A bundle's "not found" cut-offs, and the report of how its ranking scores on the tune halves.

The cut-offs come from T8 (technical spec §6.3, §7): T1 and T5 tune queries with their whole
building or object removed from the index, so any answer is wrong. Per bucket, the cut-off
is the lowest confidence that keeps at most `WRONG_TARGET` of those queries confident.
Everything here is NumPy on stored vectors: no model runs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from miroir.web.catalog import Catalog
from miroir.web.measure import BundleVectors, QuerySets, Searched, search_set
from miroir.web.routing import Bucket, Guess, PhotoType, Routing

logger = logging.getLogger(__name__)

#: Most T8 searches allowed to get a confident answer at the cut-off (the brief: about 10%).
WRONG_TARGET = 0.10
#: G5 weights from R2 (docs/results.md, "F2"): with Commons photos in the gallery the grid
#: picked EmbeddingGemma 2 at 1.0 for black-and-white building photos, 0.0 (MegaLoc alone)
#: for colour building photos and 0.2 for objects (hard switch), on T1, T2, T3 and T5 tune.
ARCHIVE_A7_WEIGHT = 1.0
COLOUR_BUILDING_A7_WEIGHT = 0.0
OBJECT_A7_WEIGHT = 0.2
#: A bucket with fewer T8 searches than this takes its cut-off from T8c (Commons photos with
#: no record) instead: T8 is drawn from archive prints and museum photos, so it holds almost
#: no colour photo of a building.
MIN_CALIBRATION_QUERIES = 20


def calibrate(
    catalog: Catalog, vectors: BundleVectors, guess: Guess, queries: QuerySets, index: str
) -> tuple[Routing, dict[str, object]]:
    """Weights from R2's G5; one cut-off per bucket from T8 (at most WRONG_TARGET confident).

    A bucket that T8 barely reaches (colour photos of buildings) takes its cut-off from T8c,
    Commons photos with no record, against T2 and T2b as the searches that should match.
    """
    draft = Routing(
        buckets=(
            Bucket(
                "archive-building", ARCHIVE_A7_WEIGHT, math.inf, PhotoType.BUILDING, greyscale=True
            ),
            Bucket(
                "colour-building",
                COLOUR_BUILDING_A7_WEIGHT,
                math.inf,
                PhotoType.BUILDING,
                greyscale=False,
            ),
            Bucket("object", OBJECT_A7_WEIGHT, math.inf, PhotoType.OBJECT),
        )
    )

    def searched(set_ids: Sequence[str]) -> list[Searched]:
        return [
            search_set(catalog, vectors, guess, draft, queries.sets[s])
            for s in set_ids
            if s in queries.sets
        ]

    pools = {
        "T8": (searched(["T8"]), searched(["T1", "T5"]), "T1+T5"),
        "T8c": (searched(["T8c"]), searched(["T2", "T2b"]), "T2+T2b"),
    }
    date = datetime.now(UTC).date().isoformat()
    buckets: list[Bucket] = []
    report: dict[str, object] = {}
    for draft_bucket in draft.buckets:
        source = "T8"
        lost_runs, found_runs, found_sets = pools[source]
        lost = _in_bucket(lost_runs, draft_bucket.name)[0]
        if lost.size < MIN_CALIBRATION_QUERIES and pools["T8c"][0]:
            source = "T8c"
            lost_runs, found_runs, found_sets = pools[source]
            lost = _in_bucket(lost_runs, draft_bucket.name)[0]
        found_conf, found_hit1 = _in_bucket(found_runs, draft_bucket.name)
        cutoff = cutoff_for(lost, WRONG_TARGET)
        matched = found_conf >= cutoff
        stats = {
            "cutoff": cutoff,
            "calibrated_on": source,
            "not_found_queries": int(lost.size),
            "not_found_confident": float((lost >= cutoff).mean()) if lost.size else 0.0,
            "in_index_sets": found_sets,
            "in_index_queries": int(found_conf.size),
            "in_index_matched": float(matched.mean()) if matched.size else 0.0,
            "matched_right_at_1": float(found_hit1[matched].mean()) if matched.any() else 0.0,
        }
        report[draft_bucket.name] = stats
        note = (
            f"A7 weight {draft_bucket.a7_weight} from R2 G5 with F2 (docs/results.md). Cut-off: "
            f"lowest confidence keeping at most {WRONG_TARGET:.0%} of {source} tune searches "
            f"confident ({stats['not_found_queries']} {draft_bucket.name} queries, index "
            f"{index}, {date}); it lets {stats['in_index_matched']:.0%} of "
            f"{stats['in_index_queries']} {found_sets} tune searches match, "
            f"{stats['matched_right_at_1']:.0%} of those right at rank 1"
        )
        buckets.append(replace(draft_bucket, cutoff=cutoff, note=note))
    routing = Routing(
        buckets=tuple(buckets),
        source=f"miroir web build-index {index}, {date}",
    )
    return routing, report


def _in_bucket(
    runs: Sequence[Searched], bucket: str
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Confidences of the searches that went to `bucket`, and whether each was right at 1."""
    conf = [r.confidence[np.asarray(r.buckets) == bucket] for r in runs]
    hit1 = [(r.ranks == 1)[np.asarray(r.buckets) == bucket] for r in runs]
    if not conf:
        return np.zeros(0), np.zeros(0, dtype=bool)
    return np.concatenate(conf), np.concatenate(hit1)


def cutoff_for(confidences: NDArray[np.float64], wrong_target: float) -> float:
    """Lowest cut-off at which at most `wrong_target` of these (wrong) searches are confident.

    A search is confident when its confidence is at or above the cut-off.
    """
    if confidences.size == 0:
        msg = "no not-found queries to set a cut-off from"
        raise ValueError(msg)
    allowed = math.floor(wrong_target * confidences.size)
    ordered = np.sort(confidences)[::-1]
    return float(np.nextafter(ordered[allowed], np.inf))


def score_sets(
    catalog: Catalog,
    vectors: BundleVectors,
    guess: Guess,
    routing: Routing,
    queries: QuerySets,
) -> dict[str, dict[str, float]]:
    """Hit rates of the site's ranking on each set's tune half, and "not found" rates."""
    out: dict[str, dict[str, float]] = {}
    cutoff = {b.name: b.cutoff for b in routing.buckets}
    for set_id, asked in queries.sets.items():
        if not asked.query_ids:
            continue
        searched = search_set(catalog, vectors, guess, routing, asked)
        confident = searched.confidence >= np.asarray([cutoff[b] for b in searched.buckets])
        row: dict[str, float] = {"queries": float(len(asked.query_ids))}
        if set_id.startswith("T8"):
            row["confident_but_wrong"] = float(confident.mean())
        else:
            scored = searched.ranks > 0
            ranks = searched.ranks[scored]
            row["scored"] = float(scored.sum())
            for k in (1, 3, 5, 10):
                row[f"hit@{k}"] = float(((ranks >= 1) & (ranks <= k)).mean())
            row["matched"] = float(confident[scored].mean())
            row["matched_right_at_1"] = (
                float((ranks[confident[scored]] == 1).mean()) if confident[scored].any() else 0.0
            )
        out[set_id] = row
        logger.info("%s: %s", set_id, ", ".join(f"{k} {v:.3f}" for k, v in row.items()))
    return out
