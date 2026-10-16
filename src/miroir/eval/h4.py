"""Step H4: load the search world, run the scenarios, score them against G5 (results.md H4).

`load_world` ranks the tune queries of T1, T2, T3 and T5 that a closed model may see with G5
(`miroir.eval.router3`, the weights picked with F2 photos) and keeps every query's A7 and A3
scores for the tools. `run` asks the model (`miroir.jobs.h4_agent`), `score` ranks what came
back, and `report` adds what it cost and how S3 used its tools.

A scenario's ranking is the model's list first, then a fallback order (S1: its own plan; S2:
S1's order; S3: its last result list, then G5). Hit@10 is the model's own top 10 whenever it
names 10; when a call fails, the query falls back to G5 and is counted as a failure.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from miroir.commons.model import is_commons_image
from miroir.config import SEED, Settings, eval_result_path
from miroir.eval.h4_tools import (
    Catalog,
    GalleryImages,
    QueryData,
    World,
    entity_infos,
)
from miroir.eval.results import SetScore, set_score, write_result
from miroir.eval.router3 import Bucket, asking_run, build_grids
from miroir.eval.run import publishable_only, search_queries
from miroir.eval.scoring import (
    Difference,
    EntityScores,
    Interval,
    ScoreResult,
    entity_scores,
    paired_bootstrap,
    per_query_values,
    rank_entities,
)
from miroir.eval.sets import load_queries
from miroir.eval.vectors import Route, RunInfo, read_vectors
from miroir.eval.vlm import H3_WEIGHT_GRID, query_text, text_gallery
from miroir.images.fetch import LONG_SIDE
from miroir.jobs.describe import parse_description
from miroir.jobs.h4_agent import (
    Caller,
    Outcome,
    Scenario,
    run_h1,
    run_s1,
    run_s2,
    run_s3,
)
from miroir.jobs.openrouter import JSON
from miroir.parallel import bounded_map
from miroir.records.store import read_records

logger = logging.getLogger(__name__)

SETS = ("T1", "T2", "T3", "T5")
GALLERY = "screening-v2-publishable-fetched+F2"
RUNS = (
    "A7-embeddinggemma-2-512-screening-v2-publishable+F2",
    "A3-megaloc-322-screening-v2-publishable+F2",
)
QUERY_RUNS = (
    "A7-embeddinggemma-2-512-T2-T2b-T8c-tune-queries",
    "A3-megaloc-322-T2-T2b-T8c-tune-queries",
)
#: G5's A7 weight per bucket, picked with F2 photos (results.md F2).
G5_WEIGHTS = {Bucket.ARCHIVE_BUILDING: 1.0, Bucket.COLOUR_BUILDING: 0.0, Bucket.OBJECT: 0.2}
ROUND = "R3"
STEP = "H"


@dataclass(frozen=True, slots=True)
class Loaded:
    world: World
    sets: Mapping[str, tuple[QueryData, ...]]
    """Every query of each set, in G5's order."""
    g5: Mapping[str, ScoreResult]
    """G5's scores on each set, as `eval route3` computes them."""
    gallery: str


def load_world(
    settings: Settings,
    *,
    gallery: str = GALLERY,
    runs: Sequence[str] = RUNS,
    query_runs: Sequence[str] = QUERY_RUNS,
    set_ids: Sequence[str] = SETS,
) -> Loaded:
    """Rank every tune query with G5 and keep both models' scores. No model runs: NumPy only."""
    grids = build_grids(
        settings,
        runs=runs,
        query_runs=query_runs,
        set_ids=set_ids,
        gallery=gallery,
        publishable_queries=True,
    )
    a7, a3 = grids.models
    if a7.gallery.image_ids != a3.gallery.image_ids:
        msg = "the two runs' galleries differ"
        raise ValueError(msg)
    image_ids = a7.gallery.image_ids
    records = read_records(settings.records_parquet, settings.images_parquet)
    entity_ids = sorted(set(a7.gallery.entity_ids))
    catalog = Catalog(entity_infos(records, entity_ids))
    text = text_gallery(records, [i for i in image_ids if not is_commons_image(i)])
    asked = [read_vectors(settings.vectors_path(r)) for r in query_runs]
    by_set: dict[str, tuple[QueryData, ...]] = {}
    query_a7: dict[str, np.ndarray] = {}
    query_a3: dict[str, np.ndarray] = {}
    g5: dict[str, ScoreResult] = {}
    for grid in grids.sets:
        loaded = publishable_only(
            settings, load_queries(settings, grid.set_id, "tune"), in_gallery=image_ids
        )
        by_id = {q.query_id: q for q in search_queries(loaded)}
        queries = [by_id[q.query_id] for q in grid.results[0].per_query]
        ids = [q.image_id for q in queries]
        ask7 = asking_run(a7.vectors, asked[0] if asked else None, ids)
        ask3 = asking_run(a3.vectors, asked[1] if asked else None, ids)
        s7 = entity_scores(ask7, a7.gallery, queries)
        s3 = entity_scores(ask3, a3.gallery, queries)
        if s7.entity_ids != catalog.entity_ids:
            msg = "entity scores and catalog disagree on entities"
            raise ValueError(msg)
        rows7, rows3 = ask7.index(), ask3.index()
        for image_id in ids:
            query_a7[image_id] = ask7.vectors[rows7[image_id]]
            query_a3[image_id] = ask3.vectors[rows3[image_id]]
        by_set[grid.set_id] = tuple(
            QueryData(grid.set_id, q, s7.values[j], s3.values[j], G5_WEIGHTS[b], b.value)
            for j, (q, b) in enumerate(zip(queries, grid.buckets, strict=True))
        )
        g5[grid.set_id] = grid.at(G5_WEIGHTS)
    columns = np.searchsorted(np.asarray(entity_ids), np.asarray(a7.gallery.entity_ids))
    images = GalleryImages(
        image_ids=image_ids,
        columns=columns.astype(np.int64),
        a7=a7.gallery.vectors,
        a3=a3.gallery.vectors,
        query_a7=query_a7,
        query_a3=query_a3,
    )
    world = World(catalog, text, images, [q for qs in by_set.values() for q in qs])
    _check_g5(world, by_set, g5)
    return Loaded(world, by_set, g5, gallery)


def _check_g5(
    world: World, sets: Mapping[str, Sequence[QueryData]], g5: Mapping[str, ScoreResult]
) -> None:
    """The tools' G5 must rank every query as `eval route3` does."""
    for set_id, queries in sets.items():
        ours = rank_rows(world, queries, [world.g5(q) for q in queries])
        theirs = {q.query_id: q.rank for q in g5[set_id].per_query}
        wrong = sum(theirs[q.query_id] != q.rank for q in ours.per_query)
        if wrong:
            msg = f"{set_id}: G5 through the tools ranks {wrong} queries differently"
            raise AssertionError(msg)


def rank_rows(
    world: World, queries: Sequence[QueryData], rows: Sequence[np.ndarray]
) -> ScoreResult:
    scores = EntityScores(
        query_ids=tuple(q.query.query_id for q in queries),
        entity_ids=world.catalog.entity_ids,
        values=np.stack(rows).astype(np.float32),
        gallery_images=len(world.images.image_ids),
    )
    return rank_entities(scores, [q.query for q in queries])


def pick(loaded: Loaded, per_set: int | None, seed: int = SEED) -> list[QueryData]:
    """Every query, or `per_set` drawn from each set (seeded), sets in turn."""
    out: list[QueryData] = []
    for set_id in sorted(loaded.sets):
        queries = list(loaded.sets[set_id])
        if per_set is not None and per_set < len(queries):
            # Shuffle then cut, so a smaller draw is part of a larger one.
            random.Random(f"{seed}-{set_id}").shuffle(queries)
            queries = sorted(queries[:per_set], key=lambda q: q.query.query_id)
        out.extend(queries)
    return out


# --- running ---------------------------------------------------------------------------------


def run(
    caller: Caller,
    world: World,
    queries: Sequence[QueryData],
    scenarios: Sequence[Scenario],
    *,
    workers: int = 4,
) -> dict[Scenario, list[Outcome]]:
    """Each query through each scenario; S2 reuses S1's saved plan. Stops on BudgetError."""
    wanted = set(scenarios)

    def one(q: QueryData) -> dict[Scenario, Outcome]:
        out: dict[Scenario, Outcome] = {}
        if Scenario.H1 in wanted:
            out[Scenario.H1] = run_h1(caller, world, q)
        if wanted & {Scenario.S1, Scenario.S2}:
            s1 = run_s1(caller, world, q)
            if Scenario.S1 in wanted:
                out[Scenario.S1] = s1
            if Scenario.S2 in wanted:
                out[Scenario.S2] = run_s2(caller, world, q, s1)
        if Scenario.S3 in wanted:
            out[Scenario.S3] = run_s3(caller, world, q)
        return out

    results: dict[Scenario, list[Outcome]] = {s: [] for s in scenarios}
    for done, by_scenario in enumerate(bounded_map(one, queries, workers=workers), start=1):
        for scenario, outcome in by_scenario.items():
            results[scenario].append(outcome)
        if done % 10 == 0 or done == len(queries):
            logger.info(
                "H4: %d/%d queries, $%.4f spent this run", done, len(queries), caller.spent_usd
            )
    return results


# --- scoring ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """One run on one set, with its paired difference against G5 on the same queries."""

    name: str
    set_id: str
    score: SetScore
    vs_g5: Mapping[str, Difference]
    """"hit@1" and "hit@10": this run minus G5."""


def score(
    loaded: Loaded, outcomes: Mapping[Scenario, Sequence[Outcome]]
) -> tuple[list[Row], dict[str, float]]:
    """Every scenario, H1 alone and H3 (A7 + H1), and G5 itself, per set.

    Returns the rows and H3's A7 weight (when H1 ran), picked by mean hit@10 over sets.
    """
    world = loaded.world
    rows: list[Row] = []
    picked = {o.query_id for os in outcomes.values() for o in os}
    for set_id in sorted(loaded.sets):
        g5 = _subset(loaded.g5[set_id], picked)
        if not g5.per_query:
            continue
        rows.append(_row(loaded, "G5", set_id, g5, g5))
        for scenario in (Scenario.S1, Scenario.S2, Scenario.S3):
            got = [o for o in outcomes.get(scenario, ()) if o.set_id == set_id]
            if got:
                result = rank_rows(
                    world, [world.queries[o.query_id] for o in got], [o.ranking for o in got]
                )
                rows.append(_row(loaded, scenario.value, set_id, result, g5))
    extra: dict[str, float] = {}
    h1 = outcomes.get(Scenario.H1, ())
    if h1:
        h1_rows, weight = _h1_rows(loaded, h1, picked)
        rows.extend(h1_rows)
        extra["h3_a7_weight"] = weight
    return rows, extra


def _h1_rows(
    loaded: Loaded, outcomes: Sequence[Outcome], picked: set[str]
) -> tuple[list[Row], float]:
    world = loaded.world
    texts = {
        o.query_id: query_text(parse_description(str(o.detail.get("answer", "")))) for o in outcomes
    }
    by_set: dict[str, list[QueryData]] = {}
    for o in outcomes:
        by_set.setdefault(o.set_id, []).append(world.queries[o.query_id])
    words = {
        q.query.query_id: world.text_scores(q, texts[q.query.query_id])
        for qs in by_set.values()
        for q in qs
    }

    def mixed(weight: float) -> dict[str, ScoreResult]:
        return {
            s: rank_rows(
                world,
                qs,
                [
                    np.where(
                        np.isfinite(world.image_scores(q, 1.0)),
                        weight * world.image_scores(q, 1.0)
                        + (1 - weight) * words[q.query.query_id],
                        -np.inf,
                    )
                    for q in qs
                ],
            )
            for s, qs in by_set.items()
        }

    tried = {
        w: float(
            np.mean([per_query_values(r.per_query)["hit@10"].mean() for r in mixed(w).values()])
        )
        for w in H3_WEIGHT_GRID
    }
    best = max(sorted(H3_WEIGHT_GRID, key=lambda w: abs(w - 0.5)), key=lambda w: tried[w])
    rows: list[Row] = []
    alone = {
        s: rank_rows(world, qs, [words[q.query.query_id] for q in qs]) for s, qs in by_set.items()
    }
    with_a7 = mixed(best)
    for set_id in sorted(by_set):
        g5 = _subset(loaded.g5[set_id], picked)
        rows.append(_row(loaded, "H1", set_id, alone[set_id], g5))
        rows.append(_row(loaded, f"H3 (A7 {best:.1f} + H1)", set_id, with_a7[set_id], g5))
    return rows, best


def _subset(result: ScoreResult, query_ids: set[str]) -> ScoreResult:
    kept = tuple(q for q in result.per_query if q.query_id in query_ids)
    return ScoreResult(
        per_query=kept,
        skipped_no_answer=0,
        gallery_images=result.gallery_images,
        gallery_entities=result.gallery_entities,
        ap_max_rank=result.ap_max_rank,
    )


def _row(loaded: Loaded, name: str, set_id: str, result: ScoreResult, g5: ScoreResult) -> Row:
    scored = set_score(result, test_set=set_id, half="tune", gallery=loaded.gallery)
    ids = {q.query_id for q in result.per_query}
    base = tuple(q for q in g5.per_query if q.query_id in ids)
    diffs = paired_bootstrap(result.per_query, base)
    return Row(
        name, set_id, scored, {d.metric: d for d in diffs if d.metric in ("hit@1", "hit@10")}
    )


# --- cost and tool use -----------------------------------------------------------------------


def cost_report(
    outcomes: Mapping[Scenario, Sequence[Outcome]], full_counts: Mapping[str, int]
) -> dict[str, JSON]:
    """Tokens and dollars per scenario (as paid, cached replies included), with the
    projection to every tune query; S3's tool use."""
    total_queries = sum(full_counts.values())
    out: dict[str, JSON] = {}
    for scenario, got in outcomes.items():
        calls = [c for o in got for c in o.calls]
        paid = sum(c.cost_usd for c in calls)
        n = max(len(got), 1)
        entry: dict[str, JSON] = {
            "queries": len(got),
            "calls": len(calls),
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "cached_tokens": sum(c.cached_tokens for c in calls),
            "cost_usd": paid,
            "usd_per_query": paid / n,
            "projected_usd": paid / n * total_queries,
            "failures": dict(Counter(o.failure.split(":")[0] for o in got if o.failure)),
        }
        if scenario is Scenario.S2:
            review = [c for o in got for c in o.calls if c.scenario == Scenario.S2.value]
            entry["review_usd_per_query"] = sum(c.cost_usd for c in review) / n
        if scenario is Scenario.S3:
            entry.update(tool_use(got))
        out[scenario.value] = entry
    return out


def tool_use(outcomes: Sequence[Outcome]) -> dict[str, JSON]:
    steps = [s for o in outcomes for s in o.steps if s.run]
    per_query = [sum(1 for s in o.steps if s.run) for o in outcomes]
    searches = [s for s in steps if s.filter_mode]
    return {
        "tool_calls_mean": float(np.mean(per_query)) if per_query else 0.0,
        "tool_calls_max": max(per_query, default=0),
        "tool_calls_per_query": dict(Counter(str(n) for n in per_query)),
        "tools": dict(Counter(s.tool for s in steps)),
        "searches_by_filter": dict(Counter(s.filter_mode for s in searches)),
        "queries_with_hard_filter": sum(
            any(s.filter_mode == "hard" for s in o.steps) for o in outcomes
        ),
        "queries_with_boost_filter": sum(
            any(s.filter_mode == "boost" for s in o.steps) for o in outcomes
        ),
        "tool_errors": sum(s.error for s in steps),
        "refused_past_budget": sum(1 for o in outcomes for s in o.steps if not s.run),
        "final_answer_called": sum(1 for o in outcomes if o.failure != "no-final-answer"),
        "final_as_text": sum(bool(o.detail.get("final_as_text")) for o in outcomes),
        "answer_size_mean": float(np.mean([len(o.final) for o in outcomes])) if outcomes else 0.0,
    }


def write_traces(path: Path, outcomes: Mapping[Scenario, Sequence[Outcome]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for got in outcomes.values():
            for o in got:
                out.write(json.dumps(o.trace(), ensure_ascii=False) + "\n")


def write_results(
    loaded: Loaded,
    rows: Sequence[Row],
    outcomes: Mapping[Scenario, Sequence[Outcome]],
    *,
    model: str,
    slug: str,
    h3_weight: float | None,
    round_id: str = ROUND,
) -> list[Path]:
    """One result file per run (G5 already has its own: results.md F2)."""
    paid = {s: sum(o.cost_usd for o in got) for s, got in outcomes.items()}
    names: dict[str, tuple[str, float, str]] = {
        "S1": (f"H4-S1-{slug}", paid.get(Scenario.S1, 0.0), "one JSON plan, carried out by code"),
        "S2": (
            f"H4-S2-{slug}",
            paid.get(Scenario.S2, 0.0),
            "S1, then the model reorders the top 20 seen as 256 px thumbnails",
        ),
        "S3": (
            f"H4-S3-{slug}",
            paid.get(Scenario.S3, 0.0),
            "the model calls image_search, text_search and view_results, up to 8 calls, "
            "then final_answer",
        ),
        "H1": (
            f"H1-{slug}",
            paid.get(Scenario.H1, 0.0),
            "the model describes the photo (H1 prompt); BM25 over record text",
        ),
    }
    if h3_weight is not None:
        names[f"H3 (A7 {h3_weight:.1f} + H1)"] = (
            f"H3-A7+H1-{slug}",
            paid.get(Scenario.H1, 0.0),
            f"A7 {h3_weight:.1f} mixed with H1's BM25 scores (weighted sum of min-max scores), "
            "weight picked by mean hit@10 over the sets",
        )
    written: list[Path] = []
    for name, (option, cost, notes) in names.items():
        scores = [r.score for r in rows if r.name == name]
        if not scores:
            continue
        info = RunInfo(
            option=option.split("-")[0] if option.startswith("H1") else option.rsplit("-", 2)[0],
            model=model,
            resolution=LONG_SIDE,
            pooling="agent",
            gallery=loaded.gallery,
            created=_today(),
            cost_usd=cost,
            wall_time_s=0.0,
            route=Route.OPENROUTER,
            notes=f"{notes}; tools over A7 and A3 vectors and BM25, gallery {loaded.gallery}",
        )
        path = eval_result_path(round_id, STEP, option)
        write_result(
            path,
            round_id=round_id,
            step=STEP,
            option=option,
            vector_run="+".join(RUNS),
            query_vector_run="+".join(QUERY_RUNS),
            vectors=info,
            scores=scores,
            today=_today(),
        )
        written.append(path)
    return written


def table(rows: Sequence[Row]) -> list[str]:
    """Lines for the console: hit@1 and hit@10 with intervals, and the paired difference."""

    def pct(v: Interval) -> str:
        return f"{100 * v.value:5.1f} [{100 * v.low:.0f}-{100 * v.high:.0f}]"

    out = [
        f"{'run':<22}{'set':<5}{'n':>5}  {'hit@1':<16}{'hit@10':<16}"
        f"{'hit@1 - G5':<22}{'hit@10 - G5'}"
    ]
    for r in rows:
        d1, d10 = r.vs_g5["hit@1"].diff, r.vs_g5["hit@10"].diff
        out.append(
            f"{r.name:<22}{r.set_id:<5}{r.score.queries:>5}  {pct(r.score.hit_at_1):<16}"
            f"{pct(r.score.hit_at_10):<16}"
            f"{100 * d1.value:+5.1f} [{100 * d1.low:+.1f},{100 * d1.high:+.1f}]  "
            f"{100 * d10.value:+5.1f} [{100 * d10.low:+.1f},{100 * d10.high:+.1f}]"
        )
    return out


def _today() -> date:
    return datetime.now(UTC).date()
