"""Steps H4 and H5 offline: the deployed agentic search, run on the T sets and scored against
the site's own photo search (technical spec §7.4, results.md H4).

The agent is the site's (`miroir.web.chat.ChatService`, its method, tools, brief and model),
unchanged. Only its two seams differ: the tools search a bundle's stored vectors
(`miroir.eval.offline`), and the model's answers are saved and replayed
(`miroir.jobs.agent_calls`). A simulated visitor (`miroir.eval.h4_visitor`) states the goal,
then answers the agent's questions from what a visitor before the answer would know.

`load` reads the bundle the site's eval steps read (`config.WEB_INDEX`) and the tune queries
of T1, T2, T3 and T5 whose photo may go to the model; `run` holds each query's conversation;
`score` ranks what came back: the cards the agent showed first, then its best-so-far, then
the site's photo ranking (the "site" row, the baseline). A conversation that ends without
results keeps that fallback and is counted as a failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from miroir.config import SEED, WEB_INDEX, Settings, eval_result_path
from miroir.eval.bm25 import fold
from miroir.eval.h4_visitor import GOAL, Visitor, VisitorFacts, visitor_facts
from miroir.eval.offline import OfflineBackend, OfflineWorld, StoredQuery, entity_order
from miroir.eval.results import SetScore, set_score, write_result
from miroir.eval.scoring import (
    Difference,
    EntityScores,
    Floats,
    Interval,
    ScoreResult,
    SearchQuery,
    paired_bootstrap,
    rank_entities,
)
from miroir.eval.sets import Query, cluster_of, load_queries
from miroir.eval.vectors import Route, RunInfo, VectorSet, read_vectors
from miroir.images.fetch import LONG_SIDE
from miroir.jobs.agent_calls import AgentKit, CachedModel, CallRecord
from miroir.jobs.openrouter import JSON
from miroir.jobs.spend import BudgetError
from miroir.records.bases import base_of
from miroir.records.model import Collection
from miroir.web.chat import (
    ChatModel,
    ChatService,
    Limits,
    MemorySpend,
    ThumbSource,
    Visit,
    VisitorInput,
)
from miroir.web.chat_method import MAX_QUESTIONS
from miroir.web.measure import COMMONS_SETS
from miroir.web.nearby import Position
from miroir.web.publishable import Publishable

logger = logging.getLogger(__name__)

SETS = ("T1", "T2", "T3", "T5")
ROUND = "R3"
STEP = "H"
#: The option label of the runs made since H4 and H5 share one agent (results files
#: `H-H5-offline-<model>.json`). H4's own files predate it.
OPTION = "H5-offline"
SLUG = "mistral-large-4"
#: Visitor messages in one conversation: the goal, then an answer to each question.
MAX_VISITS = MAX_QUESTIONS + 1
BASELINE = "site"
AGENT = "H5"
#: The trace lines the simulated visitor adds to the conversation's log.
SIMULATED = "simulated_visitor"


# --- the world -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalQuery:
    set_id: str
    query: Query
    a7: Floats = field(repr=False)
    a3: Floats = field(repr=False)
    facts: VisitorFacts

    @property
    def query_id(self) -> str:
        return self.query.query_id

    @property
    def hidden(self) -> tuple[str, ...]:
        return (self.query.query_image_id, *self.query.exclude_image_ids)

    def search_query(self) -> SearchQuery:
        q = self.query
        return SearchQuery(
            query_id=q.query_id,
            image_id=q.query_image_id,
            entity_ids=q.expected_entity_ids,
            excluded_image_ids=q.exclude_image_ids,
            cluster=cluster_of(q),
        )


@dataclass(frozen=True, slots=True)
class Loaded:
    world: OfflineWorld
    bundle: str
    sets: Mapping[str, tuple[EvalQuery, ...]]
    publishable: Publishable
    """What may go to the model: the bundle's images and the query photos that
    `PublishableImages` lets out (the rule the site's bundle file is written with)."""
    runs: tuple[str, str]
    query_runs: tuple[str, str] | None


def load(settings: Settings, *, bundle: str = WEB_INDEX, set_ids: Sequence[str] = SETS) -> Loaded:
    """The bundle as the site loads it, and every tune query it can ask. No model runs."""
    from miroir.geo.places import read_hints  # noqa: PLC0415 (reads the exports)
    from miroir.jobs.privacy import PublishableImages  # noqa: PLC0415
    from miroir.records.store import read_positions  # noqa: PLC0415
    from miroir.web.bundle import INDEXES  # noqa: PLC0415

    spec = INDEXES[bundle]
    world = OfflineWorld.load(settings.web_index_dir(bundle))
    gallery = (
        read_vectors(settings.vectors_path(spec.a7_run)),
        read_vectors(settings.vectors_path(spec.a3_run)),
    )
    query_runs = None
    commons = None
    if spec.a7_query_run and spec.a3_query_run:
        query_runs = (spec.a7_query_run, spec.a3_query_run)
        commons = (
            read_vectors(settings.vectors_path(spec.a7_query_run)),
            read_vectors(settings.vectors_path(spec.a3_query_run)),
        )
    store = PublishableImages.load(settings)
    hints = read_hints(settings)
    positions = {
        p.image_id: (p.latitude, p.longitude)
        for p in read_positions(settings.query_positions_parquet)
    }
    sets: dict[str, tuple[EvalQuery, ...]] = {}
    for set_id in set_ids:
        runs = commons if set_id in COMMONS_SETS else gallery
        if runs is None:
            logger.info("%s: no query vectors for Commons photos; not run", set_id)
            continue
        asked = load_queries(settings, set_id, "tune")
        allowed = set(store.only_publishable([q.query_image_id for q in asked]))
        sets[set_id] = eval_queries(
            set_id,
            [q for q in asked if q.query_image_id in allowed],
            runs,
            world,
            museums={r: h.museum_name for r, h in hints.items() if h.museum_name},
            positions=positions,
        )
    photos = [q.query.query_image_id for qs in sets.values() for q in qs]
    ids = [i.image_id for i in world.catalog.images]
    publishable = Publishable(store.only_publishable([*ids, *photos]))
    return Loaded(world, bundle, sets, publishable, (spec.a7_run, spec.a3_run), query_runs)


def eval_queries(
    set_id: str,
    queries: Sequence[Query],
    runs: tuple[VectorSet, VectorSet],
    world: OfflineWorld,
    *,
    museums: Mapping[str, str],
    positions: Mapping[str, tuple[float, float]],
) -> tuple[EvalQuery, ...]:
    """The queries with their stored vectors and what the visitor knows, by query id. A query
    with no vector, or whose answer the bundle does not hold, is left out (counted)."""
    catalog = world.catalog
    row7, row3 = runs[0].index(), runs[1].index()
    out: list[EvalQuery] = []
    skipped: Counter[str] = Counter()
    for q in queries:
        image = q.query_image_id
        if image not in row7 or image not in row3:
            skipped["no vector"] += 1
            continue
        entity = q.expected_entity_ids[0] if q.expected_entity_ids else ""
        row = catalog.entity_row.get(entity)
        records = catalog.entity_records(row) if row is not None else []
        if not records:
            skipped["answer not in the bundle"] += 1
            continue
        museum = next((museums[r.ref] for r in records if r.ref in museums), None)
        facts = visitor_facts(
            records,
            base_of(Collection(records[0].collection), entity),
            position=positions.get(image),
            museum=museum,
        )
        out.append(
            EvalQuery(
                set_id,
                q,
                np.asarray(runs[0].vectors[row7[image]], dtype=np.float32),
                np.asarray(runs[1].vectors[row3[image]], dtype=np.float32),
                facts,
            )
        )
    for reason, n in skipped.items():
        logger.info("%s: %d queries left out: %s", set_id, n, reason)
    return tuple(sorted(out, key=lambda e: e.query_id))


def pick(loaded: Loaded, per_set: int | None, seed: int = SEED) -> list[EvalQuery]:
    """Every query, or `per_set` drawn from each set (seeded), sets in turn."""
    out: list[EvalQuery] = []
    for set_id in sorted(loaded.sets):
        queries = list(loaded.sets[set_id])
        if per_set is not None and per_set < len(queries):
            # Shuffle then cut, so a smaller draw is part of a larger one.
            random.Random(f"{seed}-{set_id}").shuffle(queries)
            queries = sorted(queries[:per_set], key=lambda q: q.query_id)
        out.extend(queries)
    return out


#: The 10-photo test: photos per set, and how many T2 photos must carry a position.
PHOTOS = {"T1": 2, "T2": 3, "T3": 2, "T5": 3}
PHOTOS_WITH_POSITION = {"T2": 2}


def pick_photos(
    loaded: Loaded,
    counts: Mapping[str, int] = PHOTOS,
    with_position: Mapping[str, int] = PHOTOS_WITH_POSITION,
    seed: int = SEED,
) -> list[EvalQuery]:
    """A few photos per set, in the seeded order `pick` uses; some must carry a position."""
    out: list[EvalQuery] = []
    for set_id in sorted(counts):
        queries = list(loaded.sets[set_id])
        random.Random(f"{seed}-{set_id}").shuffle(queries)
        need = with_position.get(set_id, 0)
        placed = [q for q in queries if q.facts.position is not None]
        chosen = placed[:need]
        chosen += [q for q in queries if q not in chosen][: counts[set_id] - len(chosen)]
        if len(chosen) < counts[set_id] or len(placed) < need:
            msg = f"{set_id}: not enough photos for {counts[set_id]} ({need} with a position)"
            raise ValueError(msg)
        out.extend(sorted(chosen, key=lambda q: q.query_id))
    return out


# --- one conversation ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the agent did for one query."""

    query_id: str
    set_id: str
    final: tuple[str, ...]
    """The cards it showed, best first (empty when it showed none)."""
    found: str
    """What show_results said: "exact", "close", "none", or "" without results."""
    best: tuple[str, ...]
    """Its best-so-far when the conversation ended."""
    ranking: Floats = field(repr=False)
    """A score per catalog entity: `final`, then `best`, then the site's photo ranking."""
    site: Floats = field(repr=False)
    """The site's photo ranking alone: the baseline."""
    calls: tuple[CallRecord, ...]
    replayed: int
    failure: str = ""
    """Empty, or what went wrong: "no-results", "limit", "error: ..."."""
    log: tuple[Mapping[str, JSON], ...] = ()
    """The conversation's log lines, as the site writes them, and the visitor's answers."""
    visitor_given: Mapping[str, str] = field(default_factory=dict[str, str])

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def lines(self, kind: str) -> list[Mapping[str, JSON]]:
        return [line for line in self.log if line.get("type") == kind]

    def trace(self) -> dict[str, JSON]:
        """One JSON line for the run's trace file."""
        return {
            "query_id": self.query_id,
            "set_id": self.set_id,
            "final": list(self.final),
            "found": self.found,
            "best": list(self.best),
            "failure": self.failure,
            "calls": len(self.calls),
            "replayed": self.replayed,
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "cost_usd": self.cost_usd,
            "visitor_given": dict(self.visitor_given),
            "log": [dict(line) for line in self.log],
        }


class TraceLog:
    """A `miroir.web.chatlog.ChatLog` that keeps one conversation's lines in memory."""

    def __init__(self) -> None:
        self.lines: list[Mapping[str, JSON]] = []

    def write(self, day: str, conversation: str, line: Mapping[str, JSON]) -> None:
        del day, conversation
        self.lines.append(dict(line))

    async def flush(self) -> None:
        return None


async def converse(
    loaded: Loaded,
    q: EvalQuery,
    *,
    photo: bytes,
    model: CachedModel,
    fetch: Callable[[str], Awaitable[bytes]],
    limits: Limits,
    visitor_model: CachedModel | None = None,
) -> Outcome:
    """One query's conversation with the site's agent: the goal, then the visitor's answers,
    until the agent shows results, stops, or `MAX_VISITS` messages were sent."""
    world = loaded.world
    log = TraceLog()
    stored = StoredQuery(q.query_id, photo, q.a7, q.a3, q.hidden)
    catalog = world.catalog
    service = ChatService(
        backend=OfflineBackend(world, stored),
        model=model,
        thumbs=ThumbSource(loaded.publishable, fetch),
        spend=MemorySpend(),
        limits=limits,
        index_size=(len(catalog.images), catalog.entity_count),
        catalog=catalog,
        today=lambda: "eval",
        log=log,
    )
    heard_by = visitor_model or model
    visitor = Visitor(heard_by, q.facts)
    c = service.open("eval", photo=photo, words="", position=None)
    given = VisitorInput(Visit.GOAL, GOAL)
    waiting = "message"
    for _ in range(MAX_VISITS):
        after = service.send(c, given)
        assert c.task is not None
        await c.task
        stopped = model.stopped or heard_by.stopped
        if stopped is not None:
            raise BudgetError(str(stopped))
        events = c.events[after:]
        waiting = next((str(e.get("waiting")) for e in reversed(events) if e["type"] == "end"), "")
        if waiting == "answer":
            question = next(e for e in reversed(events) if e["type"] == "question")
            heard = await visitor.answer(question)
            if heard_by.stopped is not None:
                raise BudgetError(str(heard_by.stopped))
            position = None
            if heard.share_position and q.facts.position is not None:
                position = Position(*q.facts.position)
            said: dict[str, JSON] = {
                "type": SIMULATED,
                "question": question.get("text"),
                "about": question.get("about"),
                "answer": heard.text,
                "position": position is not None,
            }
            log.write("", "", said)
            given = VisitorInput(Visit.ANSWER, heard.text, position=position)
        elif waiting == "region":
            log.write("", "", {"type": SIMULATED, "region": False})
            given = VisitorInput(Visit.REGION)
        else:
            break
    calls = [*model.calls, *(heard_by.calls if heard_by is not model else ())]
    replayed = model.replayed + (heard_by.replayed if heard_by is not model else 0)
    return _outcome(
        loaded,
        q,
        stored,
        events=c.events,
        best=c.method.best,
        calls=(tuple(calls), replayed),
        log=log,
        waiting=waiting,
    )


def _outcome(
    loaded: Loaded,
    q: EvalQuery,
    stored: StoredQuery,
    *,
    events: Sequence[Mapping[str, JSON]],
    best: Sequence[str],
    calls: tuple[tuple[CallRecord, ...], int],
    log: TraceLog,
    waiting: str,
) -> Outcome:
    shown = [e for e in events if e["type"] == "results"]
    final: tuple[str, ...] = ()
    found = ""
    if shown:
        cards = shown[-1].get("cards")
        listed = cards if isinstance(cards, list) else []
        final = tuple(str(k.get("entity_id")) for k in listed if isinstance(k, dict))
        found = str(shown[-1].get("found") or "")
    errors = [str(e.get("text")) for e in events if e["type"] == "error"]
    failure = ""
    if errors:
        failure = f"error: {errors[0][:120]}"
    elif waiting == "limit":
        failure = "limit"
    elif not shown:
        failure = "no-results"
    fallback = loaded.world.site_scores(stored)
    ranking = answer_scores((*final, *best), fallback, loaded.world.catalog.entity_row)
    return Outcome(
        q.query_id,
        q.set_id,
        final,
        found,
        tuple(best),
        ranking,
        fallback,
        calls[0],
        calls[1],
        failure=failure,
        log=tuple(log.lines),
        visitor_given=q.facts.given(),
    )


def answer_scores(answer: Sequence[str], fallback: Floats, entity_row: Mapping[str, int]) -> Floats:
    """A full ranking: `answer` first in its order (an id listed twice keeps its first
    place), then every other entity by `fallback`; -inf (hidden) stays out."""
    out = np.where(np.isfinite(fallback), fallback, -np.inf).astype(np.float32)
    for i, entity_id in enumerate(dict.fromkeys(answer)):
        row = entity_row.get(entity_id)
        if row is not None:
            out[row] = 1e6 - i
    return out


# --- running ---------------------------------------------------------------------------------


def run(
    loaded: Loaded,
    queries: Sequence[EvalQuery],
    kit: AgentKit,
    *,
    limits: Limits,
    workers: int = 4,
    visitor_model: ChatModel | None = None,
) -> list[Outcome]:
    """Every query's conversation, `workers` at a time. Stops on BudgetError (every answer
    paid so far is saved: a rerun replays it)."""
    try:
        return asyncio.run(
            _run_all(loaded, queries, kit, limits=limits, workers=workers, visitor=visitor_model)
        )
    except ExceptionGroup as failed:  # the first conversation that failed stopped the others
        raise failed.exceptions[0] from None


async def _run_all(
    loaded: Loaded,
    queries: Sequence[EvalQuery],
    kit: AgentKit,
    *,
    limits: Limits,
    workers: int,
    visitor: ChatModel | None,
) -> list[Outcome]:
    gate = asyncio.Semaphore(max(workers, 1))
    done: list[Outcome] = []

    async def one(q: EvalQuery) -> Outcome:
        async with gate:
            loaded.publishable.check(q.query.query_image_id, "the model")
            photo = await asyncio.to_thread(kit.photo, q.query.query_image_id)
            model = CachedModel(kit.model, kit.cache, kit.spend, query_id=q.query_id)
            heard_by = None
            if visitor is not None:
                heard_by = CachedModel(visitor, kit.cache, kit.spend, query_id=q.query_id)
            got = await converse(
                loaded,
                q,
                photo=photo,
                model=model,
                fetch=kit.fetch,
                limits=limits,
                visitor_model=heard_by,
            )
            done.append(got)
            if len(done) % 10 == 0 or len(done) == len(queries):
                logger.info(
                    "H5 offline: %d/%d queries, $%.4f paid this run",
                    len(done),
                    len(queries),
                    kit.spend.spent_usd,
                )
            return got

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(q)) for q in queries]
    return [t.result() for t in tasks]


# --- scoring ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """One run on one set, with its paired difference against the site on the same queries."""

    name: str
    set_id: str
    score: SetScore
    vs_site: Mapping[str, Difference]
    """"hit@1" and "hit@10": this run minus the site's photo search."""


def rank_rows(loaded: Loaded, queries: Sequence[EvalQuery], rows: Sequence[Floats]) -> ScoreResult:
    """Rank each query's per-entity scores (catalog order) as every result file ranks them."""
    catalog = loaded.world.catalog
    order = entity_order(catalog)
    values = np.stack(rows).astype(np.float32)[:, order]
    scores = EntityScores(
        query_ids=tuple(q.query_id for q in queries),
        entity_ids=tuple(catalog.entity_ids[int(j)] for j in order),
        values=values,
        gallery_images=len(catalog.images),
    )
    return rank_entities(scores, [q.search_query() for q in queries])


def score(loaded: Loaded, outcomes: Sequence[Outcome]) -> list[Row]:
    """The agent and the site's photo search, per set, on the queries that ran."""
    by_id = {q.query_id: q for qs in loaded.sets.values() for q in qs}
    rows: list[Row] = []
    for set_id in sorted(loaded.sets):
        got = sorted((o for o in outcomes if o.set_id == set_id), key=lambda o: o.query_id)
        if not got:
            continue
        queries = [by_id[o.query_id] for o in got]
        site = rank_rows(loaded, queries, [o.site for o in got])
        agent = rank_rows(loaded, queries, [o.ranking for o in got])
        rows.append(_row(loaded, BASELINE, set_id, site, site))
        rows.append(_row(loaded, AGENT, set_id, agent, site))
    return rows


def _row(loaded: Loaded, name: str, set_id: str, result: ScoreResult, base: ScoreResult) -> Row:
    scored = set_score(result, test_set=set_id, half="tune", gallery=loaded.bundle)
    return Row(name, set_id, scored, _hits(result, base))


def _hits(result: ScoreResult, base: ScoreResult) -> dict[str, Difference]:
    """Paired hit@1 and hit@10 differences, result minus base, on result's queries."""
    ids = {q.query_id for q in result.per_query}
    kept = tuple(q for q in base.per_query if q.query_id in ids)
    diffs = paired_bootstrap(result.per_query, kept)
    return {d.metric: d for d in diffs if d.metric in ("hit@1", "hit@10")}


# --- cost, tool use, report ------------------------------------------------------------------


def cost_report(outcomes: Sequence[Outcome], full_counts: Mapping[str, int]) -> dict[str, JSON]:
    """Tokens and dollars (as paid, replayed answers included) with the projection to every
    tune query; how the agent used its tools and asked the visitor."""
    calls = [c for o in outcomes for c in o.calls]
    paid = sum(c.cost_usd for c in calls)
    n = max(len(outcomes), 1)
    tools = [t for o in outcomes for t in o.lines("tool")]
    per_query = [sum("refused" not in t for t in o.lines("tool")) for o in outcomes]
    asked = [v for o in outcomes for v in o.lines(SIMULATED) if "question" in v]
    return {
        "queries": len(outcomes),
        "calls": len(calls),
        "replayed": sum(o.replayed for o in outcomes),
        "prompt_tokens": sum(c.prompt_tokens for c in calls),
        "completion_tokens": sum(c.completion_tokens for c in calls),
        "cached_tokens": sum(c.cached_tokens for c in calls),
        "cost_usd": paid,
        "usd_per_query": paid / n,
        "projected_usd": paid / n * sum(full_counts.values()),
        "failures": dict(Counter(o.failure.split(":")[0] for o in outcomes if o.failure)),
        "found": dict(Counter(o.found or "-" for o in outcomes)),
        "tool_calls_mean": float(np.mean(per_query)) if per_query else 0.0,
        "tool_calls_max": max(per_query, default=0),
        "tools": dict(Counter(str(t.get("name")) for t in tools if "refused" not in t)),
        "refused": sum("refused" in t for t in tools),
        "tool_errors": sum(t.get("error") is True for t in tools),
        "questions": len(asked),
        "question_about": dict(Counter(str(v.get("about")) for v in asked)),
        "dont_know": sum("ne sais pas" in fold(str(v.get("answer"))) for v in asked),
        "position_shared": sum(v.get("position") is True for v in asked),
        "regions_declined": sum(1 for o in outcomes for v in o.lines(SIMULATED) if "region" in v),
    }


def photo_ranks(loaded: Loaded, outcomes: Sequence[Outcome]) -> dict[str, dict[str, int]]:
    """Query id, then run: the rank of the first right entity (1 is first)."""
    by_id = {q.query_id: q for qs in loaded.sets.values() for q in qs}
    got = sorted(outcomes, key=lambda o: o.query_id)
    queries = [by_id[o.query_id] for o in got]
    out: dict[str, dict[str, int]] = {o.query_id: {} for o in got}
    for name, rows in ((BASELINE, [o.site for o in got]), (AGENT, [o.ranking for o in got])):
        for q in rank_rows(loaded, queries, rows).per_query:
            out[q.query_id][name] = q.rank
    return out


def short_trace(o: Outcome) -> list[str]:
    """One conversation in a few lines: what the visitor knew, each tool call, each question
    and answer, what was shown."""
    lines = [f"{o.set_id} {o.query_id}" + (f" [{o.failure}]" if o.failure else "")]
    if o.visitor_given:
        lines.append(f"  visitor knows: {json.dumps(dict(o.visitor_given), ensure_ascii=False)}")
    for line in o.log:
        kind = line.get("type")
        if kind == "tool":
            args = json.dumps(line.get("args"), ensure_ascii=False)[:160]
            hits = line.get("hits")
            top3 = ", ".join(str(h) for h in (hits if isinstance(hits, list) else [])[:3])
            mark = " (refused)" if "refused" in line else " (error)" if line.get("error") else ""
            lines.append(f"  {line.get('name')} {args}{mark}" + (f" -> {top3}" if top3 else ""))
        elif kind == SIMULATED and "question" in line:
            shared = " (+ position)" if line.get("position") else ""
            lines.append(f"  Q: {line.get('question')}")
            lines.append(f"  A: {line.get('answer')}{shared}")
        elif kind == SIMULATED:
            lines.append("  framing asked: declined")
    lines.append(f"  shown ({o.found or 'nothing'}): {', '.join(o.final[:10]) or '-'}")
    return lines


def write_traces(path: Path, outcomes: Sequence[Outcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for o in outcomes:
            out.write(json.dumps(o.trace(), ensure_ascii=False) + "\n")


def write_results(
    loaded: Loaded, rows: Sequence[Row], outcomes: Sequence[Outcome], *, model: str
) -> Path:
    """The agent's result file (the site's own photo search has its bundle report)."""
    option = f"{OPTION}-{SLUG}"
    info = RunInfo(
        option=OPTION,
        model=model,
        resolution=LONG_SIDE,
        pooling="agent",
        gallery=loaded.bundle,
        created=_today(),
        cost_usd=sum(o.cost_usd for o in outcomes),
        wall_time_s=0.0,
        route=Route.OPENROUTER,
        notes=(
            "the site's agentic search (method, tools, brief, model) on stored vectors of "
            f"bundle {loaded.bundle}, with a simulated visitor; ranked: its cards, its "
            "best-so-far, then the site's photo ranking"
        ),
    )
    path = eval_result_path(ROUND, STEP, option)
    write_result(
        path,
        round_id=ROUND,
        step=STEP,
        option=option,
        vector_run="+".join(loaded.runs),
        query_vector_run="+".join(loaded.query_runs or ()),
        vectors=info,
        scores=[r.score for r in rows if r.name == AGENT],
        today=_today(),
    )
    return path


def table(rows: Sequence[Row]) -> list[str]:
    """Lines for the console: hit@1 and hit@10 with intervals, and the paired difference."""

    def pct(v: Interval) -> str:
        return f"{100 * v.value:5.1f} [{100 * v.low:.0f}-{100 * v.high:.0f}]"

    out = [
        f"{'run':<8}{'set':<5}{'n':>5}  {'hit@1':<16}{'hit@10':<16}"
        f"{'hit@1 - site':<22}{'hit@10 - site'}"
    ]
    for r in rows:
        d1, d10 = r.vs_site["hit@1"].diff, r.vs_site["hit@10"].diff
        out.append(
            f"{r.name:<8}{r.set_id:<5}{r.score.queries:>5}  {pct(r.score.hit_at_1):<16}"
            f"{pct(r.score.hit_at_10):<16}"
            f"{100 * d1.value:+5.1f} [{100 * d1.low:+.1f},{100 * d1.high:+.1f}]  "
            f"{100 * d10.value:+5.1f} [{100 * d10.low:+.1f},{100 * d10.high:+.1f}]"
        )
    return out


# --- the command -----------------------------------------------------------------------------


def main(
    settings: Settings,
    *,
    per_set: int,
    every_query: bool,
    photos: bool,
    label: str | None,
    max_usd: float,
    workers: int,
    key_usage_usd: float | None,
) -> list[str]:
    """`miroir vlm h4`: refuse past the budget, load, hold the conversations (paid, or
    replayed), score them, write the report (and with `every_query` the result file).
    Returns the lines for the console."""
    from miroir.jobs.factory import open_agent_kit  # noqa: PLC0415 (HTTP clients)
    from miroir.jobs.spend import Budget, SpendLedger, spent_usd  # noqa: PLC0415

    budget = Budget.from_settings(settings)
    spent = spent_usd(SpendLedger(settings.spend_ledger), key_usage_usd)
    if spent + max_usd > budget.usable_usd:
        msg = (
            f"spent ${spent:.2f}; ${max_usd:.2f} more would pass the ${budget.usable_usd:.2f} "
            "usable budget. Ask the project lead."
        )
        raise BudgetError(msg)
    loaded = load(settings)
    if every_query:
        queries = pick(loaded, None)
    elif photos:
        queries = pick_photos(loaded)
    else:
        queries = pick(loaded, per_set)
    name = label or (
        "h5-full" if every_query else "h5-photos-10" if photos else f"h5-smoke-{per_set}"
    )
    limits = Limits(settings.chat_conversation_usd, settings.chat_day_usd)
    with open_agent_kit(
        settings, run=f"{OPTION}-{name}", limit_usd=max_usd, key_usage_usd=key_usage_usd
    ) as kit:
        model = kit.model.model
        outcomes = run(loaded, queries, kit, limits=limits, workers=workers)
        paid = kit.spend.spent_usd
    rows = score(loaded, outcomes)
    counts = {s: len(qs) for s, qs in loaded.sets.items()}
    costs = cost_report(outcomes, counts)
    folder = settings.h4_report_dir(name)
    write_traces(folder / "traces.jsonl", outcomes)
    report: dict[str, JSON] = {
        "model": model,
        "bundle": loaded.bundle,
        "queries": len(queries),
        "full_counts": dict(counts),
        "new_spend_usd": paid,
        "costs": costs,
        "scores": [
            {
                "run": r.name,
                "set": r.set_id,
                "queries": r.score.queries,
                "hit@1": r.score.hit_at_1.value,
                "hit@10": r.score.hit_at_10.value,
                "hit@1 - site": r.vs_site["hit@1"].diff.value,
                "hit@10 - site": r.vs_site["hit@10"].diff.value,
            }
            for r in rows
        ],
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [f"H5 offline, {model}: {len(queries)} queries on {loaded.bundle}", *table(rows)]
    if photos and not every_query:
        ranks = photo_ranks(loaded, outcomes)
        (folder / "photos.json").write_text(json.dumps(ranks, indent=2) + "\n")
        lines.append("rank of the right answer (1 = first): site -> H5")
        lines += [f"  {q:<30}{r[BASELINE]:>6} -> {r[AGENT]}" for q, r in ranks.items()]
    short = [line for o in outcomes for line in short_trace(o)]
    (folder / "traces-short.txt").write_text("\n".join(short) + "\n", encoding="utf-8")
    lines.append(f"costs: {json.dumps(costs, ensure_ascii=False)}")
    lines.append(f"paid this run: ${paid:.4f}; traces and report in {folder}")
    if every_query:
        lines.append(f"wrote {write_results(loaded, rows, outcomes, model=model)}")
    return lines


def _today() -> date:
    return datetime.now(UTC).date()
