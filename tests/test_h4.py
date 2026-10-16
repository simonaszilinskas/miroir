"""Step H4: the tools over saved scores, and the three scenarios with a scripted model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from miroir.eval.h4_tools import (
    BOOST,
    Base,
    Catalog,
    FilterMode,
    GalleryImages,
    QueryData,
    Session,
    World,
    answer_scores,
    entity_infos,
    parse_filters,
    parse_plan,
    plan_scores,
)
from miroir.eval.scoring import SearchQuery
from miroir.eval.vlm import text_gallery
from miroir.images.fetch import Fetched
from miroir.jobs.h4_agent import (
    MAX_TOOL_CALLS,
    AgentOption,
    CallCache,
    Caller,
    ImageSource,
    Scenario,
    run_h1,
    run_s1,
    run_s2,
    run_s3,
)
from miroir.jobs.openrouter import JSON, ChatAnswer, ToolCall
from miroir.jobs.privacy import PrivacyError, PublishableImages
from miroir.jobs.spend import BudgetError, SpendLedger
from miroir.records.model import Collection, Image, ImageType, Record
from tests.fakes import jpeg

# --- a small world -------------------------------------------------------------------------------


def _record(
    ref: str,
    collection: Collection,
    entity: str,
    *,
    title: str,
    place: tuple[str, str],
    image_ids: tuple[str, ...],
    domain: tuple[str, ...] = (),
    denomination: tuple[str, ...] = (),
    publishable: bool = True,
) -> Record:
    images = tuple(
        Image(i, ref, entity, collection, f"https://x/{i}.jpg", ImageType.MAIN, n, n == 0)
        for n, i in enumerate(image_ids)
    )
    return Record(
        ref=ref,
        collection=collection,
        entity_id=entity,
        title=title,
        description="",
        denomination=denomination,
        domain=domain,
        authors=(),
        period=("roman",) if entity == "PA001" else (),
        region=None,
        departement=place[1],
        commune=place[0],
        publishable=publishable,
        pop_url="",
        linked_refs=(),
        images=images,
    )


RECORDS = (
    _record(
        "AP1",
        Collection.MEMOIRE,
        "PA001",
        title="Église Saint-Martin",
        place=("Saint-Étienne", "Loire"),
        image_ids=("i1", "i2"),
    ),
    _record(
        "AP2",
        Collection.MEMOIRE,
        "PA002",
        title="Château de Lyon",
        place=("Lyon", "Rhône"),
        image_ids=("i3",),
    ),
    _record(
        "J1",
        Collection.JOCONDE,
        "M001",
        title="",
        place=("Toulouse", "Haute-Garonne"),
        image_ids=("i4", "i5"),
        domain=("céramique",),
        denomination=("vase",),
    ),
    _record(
        "P1",
        Collection.JOCONDE,
        "M002",
        title="Secret",
        place=("Paris", "Paris"),
        image_ids=("i6",),
        publishable=False,
    ),
)
GALLERY = ("i1", "i2", "i3", "i4")
ENTITY_OF = {"i1": "PA001", "i2": "PA001", "i3": "PA002", "i4": "M001"}
VEC = {  # unit vectors in 2-D, by image
    "i1": (1.0, 0.0),
    "i2": (0.9, 0.436),
    "i3": (0.0, 1.0),
    "i4": (0.6, 0.8),
    "i5": (0.6, 0.8),
    "i6": (-1.0, 0.0),
}


def _world(query_image: str = "i5", expected: str = "M001") -> tuple[World, QueryData]:
    entity_ids = sorted(set(ENTITY_OF.values()))
    catalog = Catalog(entity_infos(RECORDS, entity_ids))  # M002 is private: not in the gallery
    query = SearchQuery("q1", query_image, (expected,))
    vectors = np.asarray([VEC[i] for i in GALLERY], dtype=np.float16)
    sims = vectors.astype(np.float32) @ np.asarray(VEC[query_image], dtype=np.float32)
    per_entity = np.full(len(entity_ids), -np.inf, dtype=np.float32)
    for image_id, sim in zip(GALLERY, sims, strict=True):
        if image_id == query_image:
            continue
        j = entity_ids.index(ENTITY_OF[image_id])
        per_entity[j] = max(per_entity[j], sim)
    a3 = per_entity[::-1].copy()  # another model, another order
    a3[~np.isfinite(per_entity)] = -np.inf
    q = QueryData("T5", query, per_entity, a3, 0.2, "object")
    images = GalleryImages(
        image_ids=GALLERY,
        columns=np.asarray([entity_ids.index(ENTITY_OF[i]) for i in GALLERY], dtype=np.int64),
        a7=vectors,
        a3=vectors,
        query_a7={query_image: np.asarray(VEC[query_image], dtype=np.float16)},
        query_a3={query_image: np.asarray(VEC[query_image], dtype=np.float16)},
    )
    text = text_gallery(RECORDS, list(GALLERY))
    return World(catalog, text, images, [q]), q


def test_image_search_mixes_the_two_models_and_hides_the_query_image() -> None:
    world, q = _world(query_image="i1", expected="PA001")
    a7_only = world.image_scores(q, 1.0)
    cols = {e: j for j, e in enumerate(world.catalog.entity_ids)}
    # i1 is hidden; PA001 still scores through i2.
    assert np.isfinite(a7_only[cols["PA001"]])
    assert a7_only[cols["PA001"]] == pytest.approx(1.0)
    mixed = world.image_scores(q, 0.5)
    assert mixed.max() <= 1.0 + 1e-6


def test_filters_boost_or_cut_and_read_names_loosely() -> None:
    world, q = _world()
    f = parse_filters({"commune": ["saint etienne"], "mode": "hard"})
    assert f.mode is FilterMode.HARD
    assert world.catalog.mask(f).tolist() == [False, True, False]  # M001 PA001 PA002
    boost = parse_filters({"base": ["Mérimée", "musée"], "domain": "église"})
    assert boost.bases == (Base.MERIMEE,)
    assert boost.ignored == ("base=musée",)
    assert boost.mode is FilterMode.BOOST
    session = Session(world, q)
    hard = session.run(
        "image_search",
        json.dumps({"model": "A3", "filters": {"base": ["joconde"], "mode": "hard"}}),
    )
    assert hard.listed == ("M001",)
    assert "1 entities match" in hard.text
    soft = session.run(
        "image_search", json.dumps({"model": "A7", "top_k": 3, "filters": {"commune": "Lyon"}})
    )
    assert len(soft.listed) == 3
    plain = world.image_scores(q, 1.0)
    col = world.catalog.column("PA002")
    assert col is not None
    assert f"{plain[col] + BOOST:.2f}" in soft.text


def test_text_search_scores_records_and_says_when_nothing_matches() -> None:
    world, q = _world()
    session = Session(world, q)
    found = session.run("text_search", json.dumps({"query": "église saint martin"}))
    assert found.listed[0] == "PA001"
    nothing = session.run("text_search", json.dumps({"query": "zzzz"}))
    assert nothing.listed == ()
    assert "no record" in nothing.text
    assert session.last_list == found.listed  # an empty search keeps the last list


def test_view_results_never_shows_a_hidden_image_and_final_answer_keeps_seen_ids() -> None:
    world, q = _world(query_image="i1", expected="PA001")
    session = Session(world, q)
    session.run("image_search", json.dumps({"model": "A7"}))
    shown = session.run("view_results", json.dumps({"entity_ids": ["PA001", "NOPE"]}))
    assert [(e, i) for _, e, i in shown.thumbnails] == [("PA001", "i2")]
    assert "unknown id" in shown.text
    final = session.run("final_answer", json.dumps({"entity_ids": ["PA001", "INVENTED"]}))
    assert final.final == ("PA001",)
    assert "1 left out" in final.text
    bad = session.run("image_search", "{not json")
    assert bad.error


def test_plan_parsing_and_its_scores() -> None:
    plan = parse_plan(
        'Here: {"image_search": {"model": "A3"}, "text_search": {"query": "vase", "weight": 2},'
        ' "filters": {"base": ["joconde"], "mode": "hard"}}'
    )
    assert plan.a7_weight == 0.0
    assert plan.text_weight == 1.0
    assert plan.filters.bases == (Base.JOCONDE,)
    assert not parse_plan("no json here").valid
    assert parse_plan('{"text_search": {"query": ""}}').text_weight == 0.0
    world, q = _world()
    _, ranking = plan_scores(world, q, plan)
    order = np.argsort(-ranking, kind="stable")
    assert world.catalog.entity_ids[order[0]] == "M001"


def test_answer_scores_put_the_answer_first_and_keep_hidden_out() -> None:
    world, q = _world()
    fallback = world.g5(q)
    hidden = ~np.isfinite(fallback)
    hidden[world.catalog.column("PA002") or 0] = True
    out = answer_scores(["PA002", "PA001"], fallback, world.catalog, hidden=hidden)
    order = [world.catalog.entity_ids[j] for j in np.argsort(-out, kind="stable")]
    assert order[0] == "PA001"
    assert out[world.catalog.column("PA002") or 0] == -np.inf


# --- a scripted model ----------------------------------------------------------------------------


@dataclass
class ScriptedModel:
    """Answers chat calls from a list; records every body sent."""

    replies: list[ChatAnswer]
    bodies: list[Mapping[str, JSON]] = field(default_factory=list)

    def chat(self, model: str, body: Mapping[str, JSON]) -> ChatAnswer:  # noqa: ARG002
        self.bodies.append(body)
        if not self.replies:
            msg = "the script ran out"
            raise AssertionError(msg)
        return self.replies.pop(0)


def _text(text: str, cost: float = 0.001) -> ChatAnswer:
    return ChatAnswer(text, cost, 1000, 50, "stop")


def _calls(*calls: tuple[str, Mapping[str, object]], cost: float = 0.001) -> ChatAnswer:
    return ChatAnswer(
        "",
        cost,
        1000,
        50,
        "tool_calls",
        tool_calls=tuple(
            ToolCall(f"abc{n:06d}", name, json.dumps(args)) for n, (name, args) in enumerate(calls)
        ),
    )


class FakeFetcher:
    def fetch(self, image: Image) -> Fetched:
        return Fetched(image.image_id, jpeg(512, 384), None)


class NoCommons:
    def fetch(self, image_id: str) -> Fetched:
        raise AssertionError(image_id)


def _caller(tmp_path: Path, model: ScriptedModel, *, limit: float = 1.0) -> Caller:
    store = PublishableImages(
        {i.image_id: i for r in RECORDS for i in r.images},
        frozenset(i.image_id for r in RECORDS if r.publishable for i in r.images),
    )
    images = ImageSource(store, FakeFetcher(), NoCommons(), "m")  # type: ignore[arg-type]
    option = AgentOption("H4-test", "test/model", "test", extra={"temperature": 0.0})
    return Caller(
        option,
        model,
        images,
        CallCache(tmp_path / "calls.jsonl"),
        SpendLedger(tmp_path / "ledger.jsonl"),
        limit_usd=limit,
    )


def test_replies_are_saved_and_replayed_without_paying(tmp_path: Path) -> None:
    world, q = _world()
    plan = '{"image_search": {"model": "A7"}, "text_search": {"query": ""}}'
    model = ScriptedModel([_text(plan)])
    first = run_s1(_caller(tmp_path, model), world, q)
    assert first.final[0] == "M001"
    assert first.replayed == 0
    sent = json.dumps(model.bodies[0])
    assert "data:image/jpeg;base64," in sent
    assert "image_ref" not in sent
    again = _caller(tmp_path, ScriptedModel([]))  # a new run reads the saved reply
    second = run_s1(again, world, q)
    assert second.replayed == 1
    assert second.final == first.final
    assert again.spent_usd == 0.0
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_a_private_image_never_reaches_the_model(tmp_path: Path) -> None:
    world, q = _world()
    model = ScriptedModel([_text("{}")])
    caller = _caller(tmp_path, model)
    with pytest.raises(PrivacyError):
        caller.check_images(["i6"])
    shown_private = QueryData("T5", SearchQuery("q2", "i6", ("M002",)), q.a7, q.a3, 0.2, "object")
    with pytest.raises(PrivacyError):
        run_h1(caller, world, shown_private)
    assert model.bodies == []


def test_spend_stops_at_the_limit(tmp_path: Path) -> None:
    world, q = _world()
    caller = _caller(tmp_path, ScriptedModel([_text("{}", cost=0.5), _text("{}")]), limit=0.4)
    run_h1(caller, world, q)
    q2 = QueryData("T5", SearchQuery("q2", "i4", ("M001",)), q.a7, q.a3, 0.2, "object")
    with pytest.raises(BudgetError):
        run_h1(caller, world, q2)


def test_s2_reorders_only_shown_candidates(tmp_path: Path) -> None:
    world, q = _world()
    model = ScriptedModel([_text('{"ranking": ["PA002", "ZZZ", "M001"]}')])
    caller = _caller(tmp_path, model)
    s1 = run_s1(
        _caller(tmp_path, ScriptedModel([_text('{"image_search": {"model": "A3"}}')])), world, q
    )
    s2 = run_s2(caller, world, q, s1)
    assert s2.final == ("PA002", "M001")
    assert s2.detail["outside_list"] == 1
    assert json.dumps(model.bodies[0]).count("data:image/jpeg") == 1 + int(str(s2.detail["shown"]))
    order = [world.catalog.entity_ids[j] for j in np.argsort(-s2.ranking, kind="stable")]
    assert order[:2] == ["PA002", "M001"]


def test_s3_runs_tools_shows_thumbnails_and_ends_on_final_answer(tmp_path: Path) -> None:
    world, q = _world()
    model = ScriptedModel(
        [
            _calls(
                ("image_search", {"model": "A3", "filters": {"base": ["joconde"], "mode": "hard"}})
            ),
            _calls(("view_results", {"entity_ids": ["M001"]})),
            _calls(("final_answer", {"entity_ids": ["M001"]})),
        ]
    )
    out = run_s3(_caller(tmp_path, model), world, q)
    assert out.final == ("M001",)
    assert out.failure == ""
    assert [s.tool for s in out.steps] == ["image_search", "view_results"]
    assert out.steps[0].filter_mode == "hard"
    third = model.bodies[2]["messages"]
    assert isinstance(third, list)
    roles = [m["role"] for m in third if isinstance(m, dict)]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "user"]
    assert "data:image/jpeg" in json.dumps(third[-1])


def test_s3_out_of_calls_is_told_to_finish_then_falls_back(tmp_path: Path) -> None:
    world, q = _world()
    search = ("text_search", {"query": "vase"})
    replies = [_calls(search) for _ in range(MAX_TOOL_CALLS)]
    replies += [_calls(search), _text("I am not sure."), _text("still"), _text("no")]
    model = ScriptedModel(replies)
    out = run_s3(_caller(tmp_path, model), world, q)
    assert out.failure == "no-final-answer"
    assert sum(s.run for s in out.steps) == MAX_TOOL_CALLS
    assert sum(not s.run for s in out.steps) == 1
    assert model.bodies[MAX_TOOL_CALLS]["tool_choice"] == "required"
    order = [world.catalog.entity_ids[j] for j in np.argsort(-out.ranking, kind="stable")]
    assert order[0] == "M001"  # its last result list


def test_h1_answer_is_kept_for_scoring(tmp_path: Path) -> None:
    world, q = _world()
    out = run_h1(_caller(tmp_path, ScriptedModel([_text('{"nom": "vase"}')])), world, q)
    assert out.detail["answer"] == '{"nom": "vase"}'
    assert out.scenario is Scenario.H1


def test_cost_report_counts_tokens_tools_and_filters(tmp_path: Path) -> None:
    from miroir.eval.h4 import cost_report  # noqa: PLC0415

    world, q = _world()
    model = ScriptedModel(
        [
            _calls(
                ("text_search", {"query": "vase", "filters": {"base": "joconde", "mode": "hard"}})
            ),
            _calls(("image_search", {"model": "A3"}), ("final_answer", {"entity_ids": ["M001"]})),
        ]
    )
    out = run_s3(_caller(tmp_path, model), world, q)
    report = cost_report({Scenario.S3: [out]}, {"T5": 10})
    s3 = report["S3"]
    assert isinstance(s3, dict)
    assert s3["calls"] == 2
    assert s3["cost_usd"] == pytest.approx(0.002)
    assert s3["projected_usd"] == pytest.approx(0.02)
    assert s3["tools"] == {"text_search": 1, "image_search": 1}
    assert s3["searches_by_filter"] == {"hard": 1, "none": 1}
    assert s3["queries_with_hard_filter"] == 1


# --- S4: ask, then search ------------------------------------------------------------------------


def test_the_visitor_never_learns_a_title_author_or_date() -> None:
    from dataclasses import fields  # noqa: PLC0415

    from miroir.eval.h4_visitor import VISITOR_FIELDS, VisitorFacts, visitor_facts  # noqa: PLC0415

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
    assert facts.commune == "Toulouse"
    assert "43.60000, 1.44000" in facts.text()
    building = visitor_facts([RECORDS[0]], Base.MERIMEE, position=None, museum="ignored")
    assert building.museum == ""
    assert "bâtiment" in building.kind


def test_near_filter_keeps_entities_placed_close_to_a_point() -> None:
    infos = entity_infos(
        RECORDS,
        ["M001", "PA001", "PA002"],
        {"PA001": (45.43, 4.39), "PA002": (45.76, 4.83)},  # Saint-Étienne, Lyon
    )
    catalog = Catalog(infos)
    f = parse_filters({"near": {"lat": 45.44, "lon": 4.39, "km": 5}, "mode": "hard"})
    assert f.near == (45.44, 4.39, 5.0)
    assert catalog.mask(f).tolist() == [False, True, False]
    assert parse_filters({"near": {"lat": "x"}}).near is None


def test_s4_asks_the_visitor_then_searches_with_the_answers(tmp_path: Path) -> None:
    from miroir.eval.h4_visitor import VisitorFacts  # noqa: PLC0415
    from miroir.jobs.h4_agent import TOOLS_S4, run_s4  # noqa: PLC0415

    world, q = _world()
    model = ScriptedModel(
        [
            _text('{"questions": ["Dans quel musée êtes-vous ?", "Qui est l\'artiste ?"]}'),
            _text('{"answers": ["Au musée Saint-Raymond, à Toulouse."]}'),
            _calls(("text_search", {"query": "vase", "filters": {"commune": ["Toulouse"]}})),
            _calls(("final_answer", {"entity_ids": ["M001"]})),
        ]
    )
    facts = VisitorFacts(commune="Toulouse", kind="un vase ou une poterie", museum="musée X")
    out = run_s4(_caller(tmp_path, model), world, q, facts)
    assert out.final == ("M001",)
    assert out.detail["questions"] == ["Dans quel musée êtes-vous ?", "Qui est l'artiste ?"]
    assert out.detail["answers"] == ["Au musée Saint-Raymond, à Toulouse.", "je ne sais pas"]
    visitor_call = json.dumps(model.bodies[1], ensure_ascii=False)
    assert "data:image" not in visitor_call  # the visitor gets words, not the photo
    assert "musée X" in visitor_call
    search_call = json.dumps(model.bodies[2], ensure_ascii=False)
    assert "Au musée Saint-Raymond" in search_call
    assert model.bodies[2]["tools"] == list(TOOLS_S4)
    assert "near" in json.dumps(TOOLS_S4)
    assert len(out.calls) == 4


def test_questions_are_sorted_by_what_they_ask() -> None:
    from miroir.eval.h4 import question_kind  # noqa: PLC0415

    assert question_kind("Où avez-vous pris la photo ?") == "place"
    assert question_kind("Dans quel musée êtes-vous ?") == "museum"
    assert question_kind("Est-ce un bâtiment ou un objet ?") == "kind"
    assert question_kind("Connaissez-vous le nom de l'artiste ?") == "name"


def test_an_id_listed_twice_keeps_its_first_place() -> None:
    world, q = _world()
    fallback = world.g5(q)
    hidden = ~np.isfinite(fallback)
    out = answer_scores(["PA002", "PA001", "M001", "PA002"], fallback, world.catalog, hidden=hidden)
    order = [world.catalog.entity_ids[j] for j in np.argsort(-out, kind="stable")]
    assert order == ["PA002", "PA001", "M001"]
