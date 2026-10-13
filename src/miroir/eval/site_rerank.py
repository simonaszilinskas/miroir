"""Step E1 inside the website: keypoint re-ranking of object searches, measured as the site runs.

The site's ranking (`miroir.web.bundle.site_search`, full index) gives each tune query its
top `SHORTLIST` entities and the best image of each. Queries the router sends to the object
bucket are matched on Modal by the website's own code (`miroir.remote.rerank`), which reads the
candidates from the image volume and computes every keypoint at search time; the time it takes
is the time a search would add. Buildings are never re-ranked.

Re-ranking is E1's `mix`: the routed mix of an entity's raw cosines plus inliers /
`MIX_INLIERS`. Entities past the shortlist keep their place. The site's search on every tune
query is kept on disk (`search_sets`); `miroir.eval.sure` reads it too.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import ConfigDict, TypeAdapter

from miroir.config import WEB_INDEX, Settings, eval_result_path
from miroir.eval.results import SetScore, set_score, write_result
from miroir.eval.scoring import QueryScore, ScoreResult, paired_bootstrap
from miroir.eval.vectors import Route, RunInfo
from miroir.eval.wire import SiteMatches, SiteMatchTask
from miroir.images.fetch import Fetched
from miroir.jobs.modal_run import PaidCall, paid_call
from miroir.jobs.modal_spec import CHUNK_TIMEOUT_S, L4
from miroir.jobs.spend import BudgetError, SpendKind
from miroir.web.bundle import SiteBundle, SiteHit, asked_sets, site_search
from miroir.web.engine import prepare_photo
from miroir.web.routing import PhotoType

logger = logging.getLogger(__name__)

#: Entities re-ranked per object search.
SHORTLIST = 30
#: E1's `mix`: this many inliers weigh as much as one unit of cosine.
MIX_INLIERS = 100.0
#: Sets measured: ranking (T3, T5, T2b), the cut-off (T8), a side check (T8c).
SETS = ("T3", "T5", "T2b", "T8", "T8c")
RANKED_SETS = ("T3", "T5", "T2b")
#: Queries per remote call.
CALL_QUERIES = 50


class SiteCalls(Protocol):
    def match_site(self, tasks: Sequence[SiteMatchTask]) -> list[SiteMatches]: ...


class SuppliesPhotos(Protocol):
    def fetch_all(self, image_ids: Iterable[str]) -> Iterable[Fetched]: ...


def matches_name(set_id: str, index: str = WEB_INDEX) -> str:
    return f"E1-site-{index}-{set_id}-top{SHORTLIST}"


_LINE = TypeAdapter(SiteMatches)


def read_site_matches(path: Path) -> dict[str, SiteMatches]:
    if not path.exists():
        return {}
    found: dict[str, SiteMatches] = {}
    for line in path.read_bytes().splitlines():
        if line.strip():
            item = _LINE.validate_json(line)
            found[item.query_id] = item
    return found


def _append(path: Path, done: Sequence[SiteMatches]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as out:
        for item in done:
            out.write(_LINE.dump_json(item) + b"\n")


@dataclass(frozen=True, slots=True)
class Searched:
    """One set's tune queries searched by the site, before re-ranking."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    set_id: str
    query_ids: tuple[str, ...]
    query_image_ids: tuple[str, ...]
    hits: tuple[SiteHit, ...]


_SEARCHED = TypeAdapter(list[Searched])


def searched_path(settings: Settings, index: str = WEB_INDEX) -> Path:
    return settings.matches_path(f"E1-site-{index}-searched")


def search_sets(settings: Settings, bundle: SiteBundle, index: str = WEB_INDEX) -> list[Searched]:
    """The site's top entities for every tune query; kept on disk (the search takes minutes)."""
    path = searched_path(settings, index)
    if path.exists():
        return _SEARCHED.validate_json(path.read_bytes())
    asked = asked_sets(settings, index)
    out: list[Searched] = []
    for set_id in SETS:
        if set_id not in asked:
            continue
        a = asked[set_id]
        hits = site_search(bundle, a.a7, a.a3, hidden=a.hidden, right=a.right, top=SHORTLIST)
        out.append(Searched(set_id, a.query_ids, a.query_image_ids, tuple(hits)))
        logger.info("%s: %d queries searched", set_id, len(hits))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_SEARCHED.dump_json(out))
    return out


def match_sets(
    settings: Settings,
    searched: Sequence[Searched],
    calls: SiteCalls,
    paid: PaidCall,
    photos: SuppliesPhotos,
    *,
    max_usd: float,
) -> float:
    """Match every object-bucket query against its shortlist on Modal; resumes. Returns cost."""
    cost = 0.0
    for s in searched:
        path = settings.matches_path(matches_name(s.set_id))
        held = read_site_matches(path)
        todo = [
            (qid, image_id, hit)
            for qid, image_id, hit in zip(s.query_ids, s.query_image_ids, s.hits, strict=True)
            if hit.photo_type is PhotoType.OBJECT and qid not in held
        ]
        logger.info("%s: %d object queries to match (%d saved)", s.set_id, len(todo), len(held))
        for chunk in batched(todo, CALL_QUERIES):
            if cost >= max_usd:
                msg = f"site re-rank: ${cost:.4f} spent, at its ${max_usd:.2f} limit; stopped"
                raise BudgetError(msg)
            tasks = _tasks(chunk, photos)
            done = paid_call(
                paid,
                lambda tasks=tasks: calls.match_site(tasks),
                machine=L4,
                timeout_s=CHUNK_TIMEOUT_S,
                kind=SpendKind.RUN,
                option="E1",
                run=matches_name(s.set_id),
                model="aliked+lightglue",
                items=len(tasks),
            )
            cost += done.cost_usd
            _append(path, done.result)
            logger.info("%s: %d queries matched, $%.4f so far", s.set_id, len(done.result), cost)
    return cost


def _tasks(
    chunk: Sequence[tuple[str, str, SiteHit]], photos: SuppliesPhotos
) -> list[SiteMatchTask]:
    commons = [image_id for _, image_id, _ in chunk if image_id.startswith("commons-")]
    jpegs = {f.image_id: f.jpeg for f in photos.fetch_all(commons)} if commons else {}
    tasks: list[SiteMatchTask] = []
    for qid, image_id, hit in chunk:
        if image_id.startswith("commons-"):
            jpeg = jpegs.get(image_id)
            if jpeg is None:
                logger.warning("%s: Commons photo not fetched; left out", qid)
                continue
            tasks.append(SiteMatchTask(qid, hit.best_images, jpeg=prepare_photo(jpeg)))
        else:
            tasks.append(SiteMatchTask(qid, hit.best_images, image_id=image_id))
    return tasks


# --- scoring ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reranked:
    """One query before and after: rank of the right entity (0 = not in the index), the
    confidence of the top answer, whether it was re-ranked."""

    query_id: str
    rank_before: int
    rank_after: int
    confidence_before: float
    confidence_after: float
    object_bucket: bool
    seconds: float | None


def rerank_hit(hit: SiteHit, matched: SiteMatches | None) -> tuple[int, float]:
    """(rank of the right entity, top confidence) after the `mix` re-ranking."""
    if matched is None or hit.photo_type is not PhotoType.OBJECT:
        return hit.right_rank, hit.confidences[0]
    if matched.candidates != hit.best_images:
        msg = f"{matched.query_id}: matches were made against another shortlist"
        raise ValueError(msg)
    keys = np.asarray(hit.confidences) + np.asarray(matched.inliers) / MIX_INLIERS
    order = np.argsort(-keys, kind="stable")
    rank = hit.right_rank
    if 1 <= rank <= len(order):
        rank = int(np.flatnonzero(order == rank - 1)[0]) + 1
    return rank, float(keys[order[0]])


def rerank_set(s: Searched, matched: Mapping[str, SiteMatches]) -> list[Reranked]:
    out: list[Reranked] = []
    for qid, hit in zip(s.query_ids, s.hits, strict=True):
        found = matched.get(qid)
        rank, confidence = rerank_hit(hit, found)
        out.append(
            Reranked(
                query_id=qid,
                rank_before=hit.right_rank,
                rank_after=rank,
                confidence_before=hit.confidences[0],
                confidence_after=confidence,
                object_bucket=hit.photo_type is PhotoType.OBJECT,
                seconds=None if found is None else found.seconds,
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class SiteRerankReport:
    seconds_median: float
    seconds_p95: float
    rows: dict[str, dict[str, float]]
    """Per ranked set, over its matched queries: hit@1/5/10 before and after."""
    hit1_difference: dict[str, tuple[float, float, float]]
    """Per ranked set: hit@1 after minus before, with its 95% paired interval."""


def matched_queries(searched: Sequence[Searched], settings: Settings) -> dict[str, list[Reranked]]:
    """Per ranked set, the object queries that were matched and have an answer in the index."""
    out: dict[str, list[Reranked]] = {}
    for s in searched:
        if s.set_id not in RANKED_SETS:
            continue
        found = read_site_matches(settings.matches_path(matches_name(s.set_id)))
        rows = [r for r in rerank_set(s, found) if r.seconds is not None and r.rank_before > 0]
        if rows:
            out[s.set_id] = rows
    return out


def report(
    searched: Sequence[Searched], settings: Settings
) -> tuple[SiteRerankReport, dict[str, list[Reranked]]]:
    sets = matched_queries(searched, settings)
    rows: dict[str, dict[str, float]] = {}
    diffs: dict[str, tuple[float, float, float]] = {}
    for set_id, found in sets.items():
        row: dict[str, float] = {"queries": float(len(found))}
        for when, ranks in (
            ("before", [r.rank_before for r in found]),
            ("after", [r.rank_after for r in found]),
        ):
            arr = np.asarray(ranks)
            for k in (1, 5, 10):
                row[f"{when}_hit@{k}"] = float((arr <= k).mean())
        rows[set_id] = row
        before = [QueryScore(r.query_id, r.rank_before, 1.0 / r.rank_before) for r in found]
        after_scores = [QueryScore(r.query_id, r.rank_after, 1.0 / r.rank_after) for r in found]
        d = next(x for x in paired_bootstrap(after_scores, before) if x.metric == "hit@1")
        diffs[set_id] = (d.diff.value, d.diff.low, d.diff.high)
    seconds = np.asarray([r.seconds for rs in sets.values() for r in rs if r.seconds is not None])
    out = SiteRerankReport(
        seconds_median=float(np.percentile(seconds, 50)) if seconds.size else 0.0,
        seconds_p95=float(np.percentile(seconds, 95)) if seconds.size else 0.0,
        rows=rows,
        hit1_difference=diffs,
    )
    return out, sets


def write_report(
    settings: Settings, bundle: SiteBundle, searched: Sequence[Searched], *, round_id: str = "R3"
) -> SiteRerankReport:
    """Two result files (the site as is, and with re-ranking) and a detail file."""
    summary, sets = report(searched, settings)
    today = datetime.now(UTC).date()
    gallery = f"web-{WEB_INDEX}"
    for option, after in (("site-base", False), ("site-mix", True)):
        scores: list[SetScore] = []
        for set_id, found in sets.items():
            ranks = [(r.query_id, r.rank_after if after else r.rank_before) for r in found]
            per_query = tuple(QueryScore(qid, rank, 1.0 / rank) for qid, rank in ranks)
            result = ScoreResult(
                per_query=per_query,
                skipped_no_answer=0,
                gallery_images=len(bundle.catalog.images),
                gallery_entities=bundle.catalog.entity_count,
                ap_max_rank=None,
            )
            scores.append(set_score(result, test_set=set_id, half="tune", gallery=gallery))
        info = RunInfo(
            option="E1" if after else "G4",
            model="site ranking" + (" + ALIKED/LightGlue on objects" if after else ""),
            resolution=512,
            pooling="site",
            gallery=gallery,
            created=today,
            cost_usd=0.0,
            wall_time_s=summary.seconds_median if after else 0.0,
            route=Route.HOSTED,
            notes=(
                f"Modal L4. Object searches re-ranked: top {SHORTLIST}, cosine mix + inliers/"
                f"{MIX_INLIERS:.0f}. Added time median {summary.seconds_median:.2f} s, p95 "
                f"{summary.seconds_p95:.2f} s. Only the object queries matched before the "
                "run was stopped (time rule failed)."
                if after
                else "The website's ranking on the full index, as its report scores it."
            ),
        )
        write_result(
            eval_result_path(round_id, "E", f"E1-{option}"),
            round_id=round_id,
            step="E",
            option=f"E1-{option}",
            vector_run=gallery,
            query_vector_run=gallery,
            vectors=info,
            scores=scores,
            today=today,
        )
    detail = eval_result_path(round_id, "E", "E1-site-mix").parent / "E-detail" / "E1-site.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(
        json.dumps(TypeAdapter(SiteRerankReport).dump_python(summary, mode="json"), indent=1)
        + "\n",
        encoding="utf-8",
    )
    return summary
