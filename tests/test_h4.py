"""Steps H4 and H5 offline: the site's agent over stored vectors, its paid calls replayed, the
simulated visitor, and the scores."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import fields, replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from miroir.eval import h4
from miroir.eval.h4_visitor import (
    DONT_KNOW,
    VISITOR_FIELDS,
    Visitor,
    VisitorFacts,
    parse_answer,
    visitor_facts,
)
from miroir.eval.offline import NoModels, OfflineBackend, OfflineError, OfflineWorld, StoredQuery
from miroir.eval.sets import Query
from miroir.eval.vectors import Route, RunInfo, VectorSet
from miroir.jobs.agent_calls import AgentKit, CachedModel, CallCache
from miroir.jobs.openrouter import JSON, ChatAnswer
from miroir.jobs.spend import Budget, BudgetError, RunSpend, SpendKind, SpendLedger
from miroir.records.bases import Base
from miroir.records.model import Collection, Record
from miroir.web.chat import Limits
from miroir.web.chat_model import ModelError
from miroir.web.chat_tools import run_tool
from miroir.web.engine import Engine, SearchRequest
from miroir.web.fake import FakeGuide, LocalBackend, NumpyGallery, ScriptedModel
from miroir.web.publishable import NotPublishableError, Publishable
from miroir.web.toolsearch import SearchKind, ToolFilters, ToolQuery
from tests import web_helpers as h

PHOTO = h.photo(90)
"""`RowEncoders` embeds it as gallery row 2 (j1, OBJ1)."""
ALL_IMAGES = ("m1", "m2", "j1", "j2", "r1", "commons-c1")


def _engines() -> tuple[Engine, OfflineWorld, h.RowEncoders]:
    """The site's engine over the fixture catalog, and the offline one over the same."""
    site, encoders = h.engine(building=False)
    offline = Engine(
        catalog=site.catalog,
        routing=site.routing,
        guess=site.guess,
        encoders=NoModels(),
        gallery=NumpyGallery(encoders.matrix),
    )
    return site, OfflineWorld(offline), encoders


def _stored(encoders: h.RowEncoders, hidden: tuple[str, ...] = ()) -> StoredQuery:
    a7, a3 = encoders.image(PHOTO)
    return StoredQuery("q1", PHOTO, a7, a3, hidden)


QUERIES = (
    ToolQuery(SearchKind.IMAGE, image=PHOTO),
    ToolQuery(SearchKind.IMAGE, image=PHOTO, a7_weight=0.0, top_k=2),
    ToolQuery(
        SearchKind.IMAGE, image=PHOTO, a7_weight=0.5, filters=ToolFilters(communes=("Rouen",))
    ),
    ToolQuery(SearchKind.TEXT, text="église vase"),
    ToolQuery(SearchKind.TEXT, text="vase", filters=ToolFilters(domains=("céramique",), hard=True)),
    ToolQuery(SearchKind.VIEW, image=PHOTO, entity_ids=("OBJ1", "PA1")),
    ToolQuery(SearchKind.LIKE, entity_ids=("PA1",), a7_weight=0.5),
    ToolQuery(SearchKind.TEXT, text="vase", filters=ToolFilters(communes=("Nulle-Part",))),
)


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.kind.value)
def test_the_offline_backend_gives_the_site_backends_tool_outputs(query: ToolQuery) -> None:
    site, world, encoders = _engines()
    offline = OfflineBackend(world, _stored(encoders))
    want = asyncio.run(LocalBackend(site).tool_search(query))
    got = asyncio.run(offline.tool_search(query))
    assert got == want
    assert want.hits or want.unknown


def test_the_offline_first_look_is_the_sites() -> None:
    site, world, encoders = _engines()
    offline = OfflineBackend(world, _stored(encoders))
    want = asyncio.run(LocalBackend(site).search(SearchRequest(image=PHOTO)))
    got = asyncio.run(offline.search(SearchRequest(image=PHOTO)))
    assert replace(got, timings_ms={}) == replace(want, timings_ms={})
    scores = world.site_scores(_stored(encoders))
    order = [site.catalog.entity_ids[int(j)] for j in np.argsort(-scores, kind="stable")]
    assert order[: len(want.cards)] == [c.entity_id for c in want.cards]


def test_hidden_images_never_score_and_never_reach_the_model() -> None:
    _, world, encoders = _engines()
    offline = OfflineBackend(world, _stored(encoders, hidden=("j1", "r1")))
    found = asyncio.run(offline.tool_search(ToolQuery(SearchKind.IMAGE, image=PHOTO, top_k=10)))
    assert "OBJ2" not in [x.entity_id for x in found.hits]  # its one image is hidden
    shown = [i.image_id for x in found.hits for i in x.images]
    assert "j1" not in shown
    assert "j2" in shown
    view = asyncio.run(
        offline.tool_search(ToolQuery(SearchKind.VIEW, image=PHOTO, entity_ids=("OBJ1", "OBJ2")))
    )
    assert [[i.image_id for i in x.images] for x in view.hits] == [["j2"]]
    first = asyncio.run(offline.search(SearchRequest(image=PHOTO)))
    assert "OBJ2" not in [c.entity_id for c in first.cards]
    scores = world.site_scores(_stored(encoders, hidden=("j1", "r1")))
    assert scores[world.catalog.entity_row["OBJ2"]] == -np.inf


def test_a_transformed_photo_is_refused_offline_and_the_model_is_told() -> None:
    _, world, encoders = _engines()
    offline = OfflineBackend(world, _stored(encoders))
    with pytest.raises(OfflineError):
        asyncio.run(offline.tool_search(ToolQuery(SearchKind.IMAGE, image=h.photo(200))))

    class Ctx:
        photo = PHOTO
        region = None
        position = None

        async def search(self, query: ToolQuery) -> object:
            return await offline.tool_search(query)

        def repeats(self, signature: object) -> None:
            del signature

        def keep_query_image(self, jpeg: bytes) -> str:
            del jpeg
            return "query-1"

        async def outlines(self) -> tuple[()]:
            return ()

    args = {"why": "Je recadre.", "transform": {"cell": "middle-center", "grayscale": True}}
    told = asyncio.run(run_tool(Ctx(), "image_search", json.dumps(args)))  # pyright: ignore[reportArgumentType]
    assert told.error
    assert "offline" in told.text


# --- paid calls --------------------------------------------------------------------------------


def _spend(tmp_path: Path, *, limit: float = 1.0, cap: float = 50.0) -> RunSpend:
    return RunSpend(
        SpendLedger(tmp_path / "ledger.jsonl"),
        Budget(cap_usd=cap, reserve_usd=0.0, probe_cap_usd=1.0),
        None,
        limit_usd=limit,
        per_item_usd=0.004,
        kind=SpendKind.RUN,
        option="H5-offline",
        run="H5-offline-test",
        model="scripted",
    )


def _said(text: str, cost: float = 0.01) -> ChatAnswer:
    return ChatAnswer(text, cost, 100, 10, "stop")


def test_answers_are_saved_paid_once_and_replayed(tmp_path: Path) -> None:
    cache = CallCache(tmp_path / "calls.jsonl")
    spend = _spend(tmp_path)
    body: dict[str, JSON] = {"messages": [{"role": "user", "content": "bonjour"}]}
    first = CachedModel(ScriptedModel([_said("salut")]), cache, spend, query_id="q1")
    assert asyncio.run(first.complete(body)).text == "salut"
    again = CachedModel(ScriptedModel([]), CallCache(cache.path), spend, query_id="q1")
    replayed = asyncio.run(again.complete(body))
    assert (replayed.text, replayed.cost_usd) == ("salut", 0.01)  # the cost it had, for the caps
    assert again.replayed == 1
    assert spend.spent_usd == pytest.approx(0.01)
    assert SpendLedger(tmp_path / "ledger.jsonl").total() == pytest.approx(0.01)


def test_the_run_limit_stops_the_run(tmp_path: Path) -> None:
    spend = _spend(tmp_path, limit=0.005)
    model = CachedModel(
        ScriptedModel([_said("a"), _said("b")]),
        CallCache(tmp_path / "c.jsonl"),
        spend,
        query_id="q",
    )
    asyncio.run(model.complete({"messages": ["1"]}))
    with pytest.raises(ModelError):
        asyncio.run(model.complete({"messages": ["2"]}))
    assert isinstance(model.stopped, BudgetError)


# --- the visitor -------------------------------------------------------------------------------


def test_the_visitor_never_learns_a_title_author_or_date() -> None:
    assert tuple(f.name for f in fields(VisitorFacts)) == VISITOR_FIELDS
    secret = Record(
        ref="SECRETREF",
        collection=Collection.JOCONDE,
        entity_id="M009",
        title="SECRETTITLE Portrait de Madame X",
        description="SECRETDESC peint en 1850",
        denomination=("SECRETDENOM", "tableau"),
        domain=("peinture",),
        authors=("SECRETAUTHOR",),
        period=("SECRETPERIOD 19e siècle",),
        region="Occitanie",
        departement="Haute-Garonne",
        commune="Toulouse",
        publishable=True,
        pop_url="https://pop/SECRETURL",
        linked_refs=("SECRETLINK",),
        images=(),
        photographers=("SECRETPHOTOGRAPHER",),
        photo_date="SECRETDATE 1972",
    )
    facts = visitor_facts(
        [secret], Base.JOCONDE, position=(43.6, 1.44), museum="musée des Augustins"
    )
    told = facts.text() + json.dumps(facts.given(), ensure_ascii=False)
    assert "SECRET" not in told
    assert "1850" not in told
    assert "1972" not in told
    assert facts.kind == "un tableau"
    assert facts.museum == "musée des Augustins"
    assert "43.60000, 1.44000" in facts.text()
    building = visitor_facts(h.catalog().records[:1], Base.MERIMEE, position=None, museum="ignored")
    assert building.museum == ""
    assert building.commune == "Toulouse"
    assert "bâtiment" in building.kind


def test_the_visitor_answers_from_its_facts_and_shares_a_position_when_offered() -> None:
    facts = VisitorFacts(commune="Toulouse", position=(43.6, 1.44), kind="un bâtiment")
    model = ScriptedModel([_said('{"answer": "À Toulouse"}'), _said("pas du JSON")])
    visitor = Visitor(model, facts)
    question: dict[str, JSON] = {"text": "Où ?", "choices": ["Ville", "Musée"], "position": True}
    heard = asyncio.run(visitor.answer(question))
    assert (heard.text, heard.share_position) == ("À Toulouse", True)
    told = json.dumps(model.bodies[0], ensure_ascii=False)
    assert "Toulouse" in told
    assert "Ville / Musée" in told
    assert asyncio.run(visitor.answer({"text": "Quand ?"})).text == DONT_KNOW
    assert parse_answer('{"answer": ""}') == DONT_KNOW


# --- a whole run ---------------------------------------------------------------------------------


def _info() -> RunInfo:
    return RunInfo("A7", "m", 512, "cls", "g", date(2026, 11, 4), 0.0, 0.0, Route.OPENROUTER)


def _loaded(encoders: h.RowEncoders, world: OfflineWorld) -> h4.Loaded:
    """T1 asks PA1 with its first photo (m1) hidden; T5 asks OBJ1 with its cover hidden."""
    ids = ("m1", "j1")
    runs = (
        VectorSet(ids, encoders.matrix[[0, 2]].astype(np.float16), _info()),
        VectorSet(ids, encoders.matrix[[0, 2]].astype(np.float16), _info()),
    )
    queries = {
        "T1": [Query("T1-1", "T1", "PA1", "m1", ("PA1",), ())],
        "T5": [Query("T5-1", "T5", "OBJ1", "j1", ("OBJ1",), ())],
    }
    sets = {
        s: h4.eval_queries(s, qs, runs, world, museums={"J1": "Musée de Rouen"}, positions={})
        for s, qs in queries.items()
    }
    return h4.Loaded(world, "fixture", sets, Publishable(ALL_IMAGES), ("a7", "a3"), None)


def _kit(tmp_path: Path, model: object) -> AgentKit:
    async def fetch(url: str) -> bytes:
        del url
        return h.photo(120)

    return AgentKit(
        model=model,  # pyright: ignore[reportArgumentType]
        cache=CallCache(tmp_path / "calls.jsonl"),
        spend=_spend(tmp_path),
        photo=lambda _image_id: PHOTO,
        fetch=fetch,
    )


class Paid:
    """FakeGuide, as if each answer cost a cent (so replays show)."""

    def __init__(self) -> None:
        self.inner = FakeGuide()
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-guide"

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        self.calls += 1
        return replace(await self.inner.complete(body), cost_usd=0.01)


def test_a_whole_offline_run_scores_the_sites_agent_and_replays_for_free(tmp_path: Path) -> None:
    _, world, encoders = _engines()
    loaded = _loaded(encoders, world)
    assert loaded.sets["T5"][0].facts.museum == "Musée de Rouen"
    queries = h4.pick(loaded, None)
    visitor = ScriptedModel([_said('{"answer": "Toulouse"}') for _ in range(4)])
    paid = Paid()
    limits = Limits(0.5, 5.0)
    outcomes = h4.run(loaded, queries, _kit(tmp_path, paid), limits=limits, visitor_model=visitor)
    assert [o.query_id for o in outcomes] == ["T1-1", "T5-1"]
    for o in outcomes:
        assert o.failure == ""
        assert o.final
        assert o.found == "close"
        assert o.lines("tool")
        asked = o.lines(h4.SIMULATED)
        assert asked
        assert asked[0]["answer"] == "Toulouse"
        hidden = loaded.sets[o.set_id][0].query.query_image_id
        assert hidden not in json.dumps([dict(line) for line in o.log])
        assert o.ranking[world.catalog.entity_row[o.final[0]]] > o.ranking.max() - 1
    rows = h4.score(loaded, outcomes)
    assert [(r.name, r.set_id) for r in rows] == [
        ("site", "T1"),
        ("H5", "T1"),
        ("site", "T5"),
        ("H5", "T5"),
    ]
    costs = h4.cost_report(outcomes, {"T1": 1, "T5": 1})
    assert costs["questions"] == 2
    assert costs["found"] == {"close": 2}
    assert h4.table(rows)
    assert h4.short_trace(outcomes[0])
    paid_calls = paid.calls
    visitor_again = ScriptedModel([])
    again = h4.run(
        loaded, queries, _kit(tmp_path, paid), limits=limits, visitor_model=visitor_again
    )
    assert paid.calls == paid_calls  # nothing asked again
    assert [o.final for o in again] == [o.final for o in outcomes]
    assert all(o.replayed == len(o.calls) for o in again)


def test_a_query_photo_that_is_not_publishable_never_leaves(tmp_path: Path) -> None:
    _, world, encoders = _engines()
    loaded = replace(_loaded(encoders, world), publishable=Publishable(("j1",)))
    with pytest.raises(NotPublishableError):
        h4.run(loaded, loaded.sets["T1"], _kit(tmp_path, Paid()), limits=Limits(0.5, 5.0))


def test_the_answer_comes_first_then_the_site_ranking() -> None:
    fallback = np.asarray([0.9, 0.1, -np.inf, 0.5], dtype=np.float32)
    rows = {"A": 0, "B": 1, "C": 2, "D": 3}
    got = h4.answer_scores(["B", "D", "B"], fallback, rows)
    assert list(np.argsort(-got, kind="stable")) == [1, 3, 0, 2]
    assert got[2] == -np.inf


def test_two_queries_with_the_same_photo_bytes_keep_their_own_vectors() -> None:
    _, world, encoders = _engines()
    first = OfflineBackend(
        world, StoredQuery("a", PHOTO, encoders.matrix[0], encoders.matrix[0], ())
    )
    second = OfflineBackend(
        world, StoredQuery("b", PHOTO, encoders.matrix[4], encoders.matrix[4], ())
    )
    query = ToolQuery(SearchKind.IMAGE, image=PHOTO, top_k=1)
    assert asyncio.run(first.tool_search(query)).hits[0].entity_id == "PA1"
    assert asyncio.run(second.tool_search(query)).hits[0].entity_id == "OBJ2"
