"""Re-ranking a vector search's top entities by keypoint matches (experiments.md step E1).

Matching runs on Modal (`miroir.remote.matching`); this module holds what goes there and
back, and re-ranks on the Mac with NumPy only. Each query's shortlist is its top entities by
cosine, each with its best image (`scoring.shortlists`). Two ways to re-rank:

- `inliers`: by the number of matches that agree on one homography (RANSAC), most first.
  Ties keep the vector order.
- `mix`: cosine plus inliers / `MIX_INLIERS`.

Entities past the shortlist keep their rank, so hit@k for k up to the shortlist length and
AP count every right entity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import ConfigDict, TypeAdapter

from miroir.eval.scoring import QueryScore, Shortlist, average_precision

#: Entities per query sent to matching.
SHORTLIST = 50
#: `mix`: this many inliers weigh as much as one unit of cosine.
MIX_INLIERS = 100.0


class Rerank(StrEnum):
    INLIERS = "inliers"
    MIX = "mix"


@dataclass(frozen=True, slots=True)
class MatchTask:
    """One query image and the candidate images to match it against."""

    query_id: str
    image_id: str
    candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    __pydantic_config__ = ConfigDict(extra="forbid")

    image_id: str
    matches: int
    """LightGlue matches."""
    inliers: int
    """Matches that agree on one homography (RANSAC)."""
    inliers_f: int
    """Matches that agree on one fundamental matrix (RANSAC); kept for later study."""


@dataclass(frozen=True, slots=True)
class QueryMatches:
    """What matching one query cost and found. Times are on the Modal GPU, in seconds."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    query_id: str
    image_id: str
    candidates: tuple[CandidateMatch, ...]
    query_s: float
    """Keypoints of the query image: paid at search time."""
    match_s: float
    """Matching and RANSAC against every candidate: paid at search time."""
    candidates_s: float
    """Keypoints of the candidates not already held: done ahead of time in a real index."""

    @property
    def search_s(self) -> float:
        """Time a search would add: the gallery's keypoints are computed ahead of time."""
        return self.query_s + self.match_s


@dataclass(frozen=True, slots=True)
class MatchChunk:
    """What one remote call sends back."""

    queries: tuple[QueryMatches, ...]
    load_s: float
    missing: tuple[str, ...]
    """Images the call could not read; their pairs are left out of `queries`."""


_LINE = TypeAdapter(QueryMatches)


def tasks(lists: Iterable[Shortlist]) -> list[MatchTask]:
    return [MatchTask(s.query_id, s.image_id, s.image_ids) for s in lists]


def append_matches(path: Path, done: Sequence[QueryMatches]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as out:
        for line in done:
            out.write(_LINE.dump_json(line) + b"\n")


def read_matches(path: Path) -> dict[str, QueryMatches]:
    """Matches saved so far, by query id. A missing file holds none."""
    if not path.exists():
        return {}
    found: dict[str, QueryMatches] = {}
    for line in path.read_bytes().splitlines():
        if line.strip():
            item = _LINE.validate_json(line)
            found[item.query_id] = item
    return found


def rerank(shortlist: Shortlist, matched: QueryMatches, how: Rerank) -> QueryScore:
    """The query's score once its shortlist is re-ranked by `how`."""
    inliers: Mapping[str, int] = {c.image_id: c.inliers for c in matched.candidates}
    missing = [i for i in shortlist.image_ids if i not in inliers]
    if missing:
        msg = f"{shortlist.query_id}: no matches for {len(missing)} candidate(s), e.g. {missing[0]}"
        raise ValueError(msg)
    counts = np.asarray([inliers[i] for i in shortlist.image_ids], dtype=np.float64)
    keys = counts if how is Rerank.INLIERS else np.asarray(shortlist.scores) + counts / MIX_INLIERS
    order = np.argsort(-keys, kind="stable")  # ties keep the vector order
    new_rank = {shortlist.entity_ids[int(i)]: r + 1 for r, i in enumerate(order)}
    ranks = [new_rank.get(entity, rank) for entity, rank in shortlist.right_ranks]
    return QueryScore(shortlist.query_id, rank=min(ranks), ap=average_precision(ranks))


def search_times(matched: Iterable[QueryMatches]) -> tuple[float, float]:
    """Median and 95th percentile of the time re-ranking adds to one search, in seconds."""
    times = np.asarray([m.search_s for m in matched], dtype=np.float64)
    if times.size == 0:
        msg = "no matched queries"
        raise ValueError(msg)
    median, p95 = np.percentile(times, [50, 95])
    return float(median), float(p95)
