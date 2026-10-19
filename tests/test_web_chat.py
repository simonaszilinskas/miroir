"""The conversation page (step H5): a scripted model drives the tools over the tiny index."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from miroir.jobs.openrouter import JSON, ChatAnswer, ToolCall
from miroir.outlines import Outlines
from miroir.records.bases import place_key
from miroir.web import chat as chat_module
from miroir.web.app import create_app
from miroir.web.chat import (
    BRIEF,
    MAX_TOOL_CALLS,
    MAX_VIEWS,
    ChatService,
    Limits,
    MemorySpend,
    ThumbSource,
)
from miroir.web.chat_tools import TOOLS, ToolName, tools_for
from miroir.web.chatlog import ChatLog
from miroir.web.engine import SearchRequest, SearchResponse
from miroir.web.fake import FakeGuide, LocalBackend, ScriptedModel
from miroir.web.publishable import (
    PUBLISHABLE_FILE,
    NotPublishableError,
    Publishable,
    read_publishable,
    write_publishable,
)
from miroir.web.toolsearch import (
    BOOST,
    Near,
    SearchKind,
    ToolFilters,
    ToolHits,
    ToolQuery,
    ToolSearch,
)
from tests import web_helpers as h

PASSWORD = "test-mot-de-passe"
PHOTO = base64.b64encode(h.photo(90)).decode()
EXACT = "Je cherche ce monument précis."
SIMILAR = "Je cherche des objets semblables."
CAPS = Limits(0.10, 2.0)
WHY = "Le dernier résultat ne tranche pas ; je vérifie."


def call(name: str, args: Mapping[str, JSON], cost: float = 0.001, text: str = "") -> ChatAnswer:
    return calls([(name, args)], cost, text)


def calls(
    pairs: Sequence[tuple[str, Mapping[str, JSON]]], cost: float = 0.001, text: str = ""
) -> ChatAnswer:
    return ChatAnswer(
        text=text,
        cost_usd=cost,
        prompt_tokens=100,
        completion_tokens=10,
        finish_reason="tool_calls",
        tool_calls=tuple(
            ToolCall(f"call{n:05d}", name, json.dumps(args)) for n, (name, args) in enumerate(pairs)
        ),
    )


class CountingBackend:
    """The local backend, counting the searches that reach it. A `cutoff` above 1 keeps the
    first look from rating anything "Très proche" (no sure match)."""

    def __init__(self, cutoff: float = 0.9) -> None:
        engine, _ = h.engine(building=True, cutoff=cutoff)
        self.engine = engine
        self.local = LocalBackend(engine)
        self.counts = {"search": 0, "tool": 0}
        self.queries: list[ToolQuery] = []

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.counts["search"] += 1
        return await self.local.search(request)

    async def tool_search(self, query: ToolQuery) -> ToolHits:
        self.counts["tool"] += 1
        self.queries.append(query)
        return await self.local.tool_search(query)

    async def warm(self) -> None:
        return None

    async def detect(self, jpeg: bytes) -> Outlines:
        return await self.local.detect(jpeg)


class Fetches:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return h.photo(120)


class Rig:
    """An app with a scripted model; `start`, `send` and `read` talk to it as the page does."""

    def __init__(
        self,
        answers: Sequence[ChatAnswer],
        *,
        publishable: Sequence[str] = ("m1", "m2", "j1", "j2", "r1", "commons-c1"),
        limits: Limits = CAPS,
        spend: MemorySpend | None = None,
        cutoff: float = 0.9,
        log: ChatLog | None = None,
    ) -> None:
        self.model = ScriptedModel(answers)
        self.backend = CountingBackend(cutoff)
        self.fetches = Fetches()
        self.spend = spend or MemorySpend()
        self.service: ChatService | None = None

        def chat(fetch: object) -> ChatService:
            del fetch
            self.service = ChatService(
                backend=self.backend,
                model=self.model,
                thumbs=ThumbSource(Publishable(publishable), self.fetches),
                spend=self.spend,
                limits=limits,
                index_size=(6, 3),
                today=lambda: "2026-10-16",
                log=log,
            )
            return self.service

        app = create_app(
            catalog=self.backend.engine.catalog,
            backend=self.backend,
            password=PASSWORD,
            session_key=b"k" * 32,
            secure_cookies=False,
            chat=chat,
        )
        self.client = TestClient(app)

    def __enter__(self) -> Rig:
        self.client.__enter__()
        assert self.client.post("/api/login", json={"password": PASSWORD}).status_code == 200
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.__exit__(None, None, None)

    def start(self, *, photo: bool = True, words: str = "") -> str:
        body: dict[str, object] = {"text": words}
        if photo:
            body["image"] = PHOTO
        response = self.client.post("/api/chat/open", json=body)
        assert response.status_code == 200, response.text
        return response.json()["conversation_id"]

    def send(self, cid: str, kind: str, text: str = "", **extra: object) -> list[dict[str, JSON]]:
        response = self.client.post(
            "/api/chat/send", json={"conversation_id": cid, "kind": kind, "text": text, **extra}
        )
        assert response.status_code == 200, response.text
        return self.read(cid, response.json()["after"])

    def read(self, cid: str, after: int) -> list[dict[str, JSON]]:
        events: list[dict[str, JSON]] = []
        while True:
            response = self.client.post(
                "/api/chat/events", json={"conversation_id": cid, "after": after}
            )
            lines = [json.loads(line) for line in response.text.splitlines() if line]
            events += [e for e in lines if e["type"] not in ("ping", "continue")]
            if not lines or lines[-1]["type"] != "continue":
                return events
            after = int(lines[-1]["after"])


def searching(name: str, args: Mapping[str, JSON], **kw: object) -> ChatAnswer:
    """A search or a look, with its why."""
    return call(name, {**args, "why": WHY}, **kw)  # pyright: ignore[reportArgumentType]


def shown(*ids: str, found: str = "close", cost: float = 0.001) -> ChatAnswer:
    results: list[JSON] = [{"id": i, "reason": "r"} for i in ids]
    summary = "" if found == "exact" else "Je ne trouve pas cet objet précis. Les plus proches."
    return call("show_results", {"found": found, "summary": summary, "results": results}, cost)


#: Looks at the first look's top results, as the model must before a search.
LOOK = searching("view_results", {"entity_ids": ["OBJ1", "PA1"]})


def of(events: Sequence[dict[str, JSON]], kind: str) -> list[dict[str, JSON]]:
    return [e for e in events if e["type"] == kind]


def card_ids(events: Sequence[dict[str, JSON]]) -> list[str]:
    found = of(events, "results")[0]["cards"]
    assert isinstance(found, list)
    return [str(c["entity_id"]) for c in found if isinstance(c, dict)]


def tool_messages(body: Mapping[str, JSON]) -> list[str]:
    messages = body["messages"]
    assert isinstance(messages, list)
    return [str(m["content"]) for m in messages if isinstance(m, dict) and m["role"] == "tool"]


def body_text(body: Mapping[str, JSON]) -> str:
    return json.dumps(body["messages"], ensure_ascii=False)


# --- the flow --------------------------------------------------------------------------------


def test_question_then_answer_then_search_then_results() -> None:
    script = [
        call(
            "ask_visitor",
            {"question": "Où est ce bâtiment ?", "choices": ["Dans une ville", "À la campagne"]},
            text="Une question d’abord.",
        ),
        LOOK,
        searching("image_search", {"model": "auto", "filters": {"commune": ["Toulouse"]}}),
        call(
            "show_results",
            {
                "found": "close",
                "summary": "Je ne trouve pas ce bâtiment précis.",
                "results": [{"id": "PA1", "reason": "même façade"}],
            },
        ),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        first = rig.send(cid, "goal", EXACT)
        assert of(first, "say")[0]["text"] == "Une question d’abord."
        question = of(first, "question")[0]
        assert question["choices"] == ["Dans une ville", "À la campagne"]
        assert first[-1] == {**first[-1], "type": "end", "waiting": "answer"}
        look = next(s for s in of(first, "step") if s["state"] == "done")
        assert "Première recherche" in str(look["label"])

        second = rig.send(cid, "answer", "À Toulouse")
        steps = [s for s in of(second, "step") if s["state"] == "done"]
        assert "Regarde des images" in str(steps[0]["label"])
        assert "Recherche par image" in str(steps[1]["label"])
        assert "Toulouse" in str(steps[1]["label"])
        assert steps[1]["why"] == WHY
        assert card_ids(second) == ["PA1"]
        results = of(second, "results")[0]["cards"]
        assert isinstance(results, list)
        assert isinstance(results[0], dict)
        assert results[0]["reason"] == "même façade"
        assert second[-1]["waiting"] == "message"

    opening = rig.model.bodies[0]
    assert "data:image/jpeg;base64," in body_text(opening)
    assert "First look" in body_text(opening)
    assert "The visitor answers: «À Toulouse»" in tool_messages(rig.model.bodies[1])[0]
    assert rig.backend.counts["search"] == 1


@pytest.mark.parametrize("goal", [EXACT, SIMILAR])
def test_the_visitors_goal_is_the_first_thing_the_model_reads(goal: str) -> None:
    with Rig([shown("PA1")]) as rig:
        cid = rig.start()
        rig.send(cid, "goal", goal)
    messages = rig.model.bodies[0]["messages"]
    assert isinstance(messages, list)
    system, opening = messages[0], messages[1]
    assert isinstance(system, dict)
    assert isinstance(opening, dict)
    assert "One exact building or object" in str(system["content"])
    assert "Similar things" in str(system["content"])
    parts = opening["content"]
    assert isinstance(parts, list)
    assert isinstance(parts[0], dict)
    assert parts[0]["text"] == f"The visitor's goal, in their own words: «{goal}»"


def test_a_start_with_words_only_offers_no_image_tools() -> None:
    script = [
        searching("text_search", {"query": "église Saint-Pierre"}),
        shown("PA1"),
    ]
    with Rig(script) as rig:
        cid = rig.start(photo=False, words="église Saint-Pierre à Toulouse")
        events = rig.send(cid, "goal", EXACT)
    assert rig.backend.counts["search"] == 0
    names = {
        str(t["function"]["name"]) for t in json.loads(json.dumps(rig.model.bodies[0]["tools"]))
    }
    assert ToolName.IMAGE_SEARCH.value not in names
    assert ToolName.ASK_REGION.value not in names
    assert "typed words: «église Saint-Pierre à Toulouse»" in body_text(rig.model.bodies[0])
    assert "data:image" not in body_text(rig.model.bodies[0])
    assert card_ids(events) == ["PA1"]


def test_a_visitor_message_after_results_starts_a_new_search() -> None:
    script = [
        shown("PA1"),
        searching("text_search", {"query": "vase céramique"}),
        shown("OBJ1"),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
        events = rig.send(cid, "message", "Pas celui-ci, c’est un vase")
    assert card_ids(events) == ["OBJ1"]
    assert "The visitor writes: «Pas celui-ci, c’est un vase»" in body_text(rig.model.bodies[1])


def test_a_framed_region_comes_back_as_an_image_to_search() -> None:
    script = [
        call("ask_region", {"reason": "Quel tableau ?"}),
        LOOK,
        searching("image_search", {"source": "region", "model": "A3"}),
        shown("OBJ1"),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        first = rig.send(cid, "goal", EXACT)
        assert of(first, "region")[0]["text"] == "Quel tableau ?"
        assert first[-1]["waiting"] == "region"
        crop = base64.b64encode(h.photo(130)).decode()
        second = rig.send(cid, "region", region=crop)
    assert of(second, "results")
    assert "The visitor framed a part" in tool_messages(rig.model.bodies[1])[0]
    assert "sur la partie encadrée" in str(of(second, "step")[-1]["label"])


# --- what H4 taught --------------------------------------------------------------------------


def test_an_unknown_filter_key_is_an_error_naming_the_valid_ones() -> None:
    script = [
        LOOK,
        searching("image_search", {"filters": {"region": ["Bretagne"]}}),
        searching("image_search", {"filters": {"base": ["museum"]}}),
        shown("PA1"),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
    _, first, second = tool_messages(rig.model.bodies[3])
    assert "unknown filter key(s): region" in first
    assert "Valid keys: base, departement, commune, domain, period, near, mode" in first
    assert "unknown base 'museum'" in second
    assert "merimee, palissy, joconde, memoire, rose-valland" in second
    assert rig.backend.counts["tool"] == 1  # the look only


def test_a_repeated_search_is_not_run_and_the_model_is_told() -> None:
    script = [
        LOOK,
        searching("text_search", {"query": "église Saint-Pierre Toulouse"}),
        searching("text_search", {"query": "Toulouse, église de Saint-Pierre"}),
        searching(
            "text_search", {"query": "église Saint-Pierre", "filters": {"commune": "Toulouse"}}
        ),
        shown("PA1"),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
    replies = tool_messages(rig.model.bodies[4])
    assert replies[2].startswith("not run: this repeats step 3")
    assert not replies[3].startswith("not run")  # other filters: a new search
    assert rig.backend.counts["tool"] == 3  # the look, then two searches


def test_searches_stop_at_the_cap_for_one_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_module, "MAX_STALE", 99)  # the tiny index has nothing new to add
    words = [
        "vase",
        "église",
        "portail",
        "nature",
        "morte",
        "lièvre",
        "céramique",
        "toulouse",
        "rouen",
    ]
    script = [
        LOOK,
        calls([("text_search", {"query": w, "why": WHY}) for w in words]),
        shown("PA1"),
    ]
    with Rig(script) as rig:  # the first look is sure: only the cap of 8 stops searches
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
    replies = tool_messages(rig.model.bodies[2])[1:]
    assert len(replies) == len(words)
    assert sum(r.startswith("not run: no search") for r in replies) == len(words) - MAX_TOOL_CALLS
    assert rig.model.bodies[2]["tool_choice"] == "required"


def test_ids_the_model_never_saw_are_dropped() -> None:
    script = [
        searching("text_search", {"query": "église"}),
        call(
            "show_results",
            {
                "found": "close",
                "summary": "Les plus proches.",
                "results": [
                    {"id": "OBJ2", "reason": "jamais vu"},
                    {"id": "PA1", "reason": "vu"},
                    {"id": "PA999", "reason": "inventé"},
                ],
            },
        ),
    ]
    with Rig(script) as rig:
        cid = rig.start(photo=False, words="église")
        events = rig.send(cid, "goal", EXACT)
    assert card_ids(events) == ["PA1"]


def test_thumbnails_of_records_that_are_not_publishable_never_leave() -> None:
    script = [
        searching("view_results", {"entity_ids": ["PA1", "OBJ1", "OBJ2"]}),
        shown("OBJ1"),
    ]
    with Rig(script, publishable=("j2",)) as rig:  # only OBJ1's back may go out
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
    assert rig.fetches.urls == ["https://example.org/j2.jpg"]
    sent = body_text(rig.model.bodies[1])
    assert sent.count("data:image/jpeg") == 2  # the visitor's photo and j2
    view = tool_messages(rig.model.bodies[1])[0]
    assert "PA1 | merimee" in view
    assert "(no image may be shown)" in view
    assert rig.service is not None
    with pytest.raises(NotPublishableError):
        rig.service.thumbs.publishable.check("m1", "the model")


def test_a_thumbnail_source_refuses_an_image_that_is_not_publishable() -> None:
    fetches = Fetches()
    source = ThumbSource(Publishable(["j1"]), fetches)
    with pytest.raises(NotPublishableError):
        asyncio.run(source.jpeg("m1", "https://example.org/m1.jpg"))
    assert fetches.urls == []
    assert asyncio.run(source.jpeg("j1", "https://example.org/j1.jpg")) is not None


# --- money -----------------------------------------------------------------------------------


def test_the_conversation_cap_stops_the_model() -> None:
    script = [
        searching("text_search", {"query": "église"}, cost=0.06),
        shown("PA1", cost=0.06),
    ]
    with Rig(script, limits=Limits(0.10, 2.0)) as rig:
        cid = rig.start()
        events = rig.send(cid, "goal", EXACT)
    assert len(rig.model.bodies) == 1
    limit = of(events, "limit")[0]
    assert "limite de dépense (0,10 $)" in str(limit["text"])
    assert events[-1]["waiting"] == "limit"


def test_the_day_cap_stops_the_model_before_any_call() -> None:
    spend = MemorySpend()
    with Rig([], spend=spend, limits=Limits(0.10, 2.0)) as rig:
        asyncio.run(spend.add("2026-10-16", 1.995))
        cid = rig.start()
        events = rig.send(cid, "goal", EXACT)
    assert rig.model.bodies == []
    assert "pour aujourd’hui" in str(of(events, "limit")[0]["text"])


def test_spending_adds_up_per_day() -> None:
    spend = MemorySpend()
    script = [shown("PA1", cost=0.02)]
    with Rig(script, spend=spend) as rig:
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
        assert asyncio.run(spend.spent("2026-10-16")) == pytest.approx(0.02)


# --- the page and the tools ------------------------------------------------------------------


def test_the_chat_address_opens_the_conversation_view() -> None:
    with Rig([]) as rig:
        response = rig.client.get("/chat", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/#/conversation"
        assert rig.client.get("/api/chat/status").json() == {"enabled": True}
        rig.client.cookies.clear()
        assert rig.client.post("/api/chat/open", json={"text": "x"}).status_code == 401


def test_without_a_model_the_chat_says_it_is_off() -> None:
    engine, _ = h.engine()
    app = create_app(
        catalog=engine.catalog,
        backend=LocalBackend(engine),
        password=PASSWORD,
        session_key=b"k" * 32,
        secure_cookies=False,
    )
    client = TestClient(app)
    client.post("/api/login", json={"password": PASSWORD})
    assert client.get("/api/chat/status").json() == {"enabled": False}
    assert client.post("/api/chat/open", json={"text": "église"}).status_code == 503


def test_the_fake_guide_runs_a_whole_conversation() -> None:
    with Rig([]) as rig:
        assert rig.service is not None
        rig.service.model = FakeGuide()
        cid = rig.start()
        first = rig.send(cid, "goal", SIMILAR)
        assert of(first, "question")
        second = rig.send(cid, "answer", "Un bâtiment")
    assert card_ids(second)


def test_every_tool_has_a_unique_name_and_a_json_schema() -> None:
    names = [t.name.value for t in TOOLS]
    assert len(names) == len(set(names))
    for tool in TOOLS:
        schema = tool.schema()
        assert json.loads(json.dumps(schema)) == schema
        assert tool.description
    assert {t.name for t in tools_for(photo=False)} == set(ToolName) - {
        ToolName.IMAGE_SEARCH,
        ToolName.ASK_REGION,
    }


def test_filters_favour_by_default_and_cut_only_when_hard() -> None:
    engine, _ = h.engine(building=True)
    search = ToolSearch(engine)
    words = ToolQuery(SearchKind.TEXT, text="vase église nature")
    plain = {x.entity_id: x.score for x in search.run(words).hits}
    soft = search.run(
        ToolQuery(SearchKind.TEXT, text=words.text, filters=ToolFilters(communes=("toulouse",)))
    )
    assert {x.entity_id: x.score for x in soft.hits}["PA1"] == pytest.approx(plain["PA1"] + BOOST)
    assert len(soft.hits) == len(plain)
    assert soft.matching == 1
    hard = search.run(
        ToolQuery(
            SearchKind.TEXT,
            text=words.text,
            filters=ToolFilters(departements=("Seine Maritime",), hard=True),
        )
    )
    assert [x.entity_id for x in hard.hits] == ["OBJ1"]
    by_period = search.mask(ToolFilters(periods=("18e siècle",)))
    assert [engine.catalog.entity_ids[i] for i, k in enumerate(by_period) if k] == ["OBJ1"]


def test_a_text_search_with_no_known_word_says_so() -> None:
    engine, _ = h.engine()
    found = ToolSearch(engine).run(ToolQuery(SearchKind.TEXT, text="zzzz"))
    assert found.words_found is False
    assert found.hits == ()


def test_near_without_places_is_reported() -> None:
    engine, _ = h.engine()
    near = ToolFilters(near=Near(43.6, 1.44, 2.0))
    found = ToolSearch(engine).run(ToolQuery(SearchKind.TEXT, text="église", filters=near))
    assert found.near_ignored


def test_view_lists_images_closest_first_with_the_cover_for_words() -> None:
    engine, _ = h.engine()
    found = ToolSearch(engine).run(ToolQuery(SearchKind.VIEW, entity_ids=("OBJ1", "nope")))
    assert [x.entity_id for x in found.hits] == ["OBJ1"]
    assert [i.image_id for i in found.hits[0].images] == ["j1", "j2"]


def test_the_brief_names_the_tool_cap() -> None:
    assert "{calls}" in BRIEF


def test_the_publishable_file_round_trips_and_a_missing_one_lets_nothing_out(
    tmp_path: Path,
) -> None:
    path = tmp_path / PUBLISHABLE_FILE
    assert write_publishable(path, ["j1", "m1", "j1"]) == 2
    found = read_publishable(path)
    assert found.allows("j1")
    assert not found.allows("r1")
    assert len(read_publishable(tmp_path / "missing.txt.gz")) == 0


@pytest.mark.parametrize(
    ("pop", "said"),
    [
        ("Ferté-Bernard (La)", "La Ferté-Bernard"),
        ("Isle-Adam (L')", "L’Isle-Adam"),
        ("Mans (Le)", "le mans"),
        ("Saint-Étienne", "saint etienne"),
    ],
)
def test_a_commune_matches_however_its_article_is_placed(pop: str, said: str) -> None:
    assert place_key(pop) == place_key(said)


def say_only(text: str) -> ChatAnswer:
    return ChatAnswer(text, 0.001, 10, 5, "stop")


def test_a_reply_without_a_tool_call_is_nudged_once() -> None:
    script = [
        say_only("Je relance avec une recherche par mots."),
        shown("PA1"),
        say_only("Merci !"),
        say_only("Bonne visite."),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        first = rig.send(cid, "goal", EXACT)
        assert card_ids(first) == ["PA1"]
        second = rig.send(cid, "message", "Merci")
    assert "You wrote without calling a tool" in body_text(rig.model.bodies[1])
    assert [e["text"] for e in of(second, "say")] == ["Merci !", "Bonne visite."]
    assert second[-1]["waiting"] == "message"
    assert len(rig.model.bodies) == 4


def test_looks_at_thumbnails_have_their_own_cap() -> None:
    views = [searching("view_results", {"entity_ids": [e]}) for e in ("PA1", "OBJ1", "OBJ2")]
    script = [
        *views,
        searching("view_results", {"entity_ids": ["OBJ1", "PA1"]}),
        shown("PA1"),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
    last = tool_messages(rig.model.bodies[-1])
    assert last[-1].startswith(f"not run: {MAX_VIEWS} looks at thumbnails")


def test_the_home_page_offers_both_searches() -> None:
    with Rig([]) as rig:
        page = rig.client.get("/").text
        assert 'data-view="conversation"' in page
        assert "Recherche simple" in page
        assert "Recherche agentique" in page
        assert 'name="mode" value="agent"' in page
        assert "Votre photo est envoyée à ce modèle" in page
        script = rig.client.get("/static/chat.js")
        assert script.status_code == 200
        assert "GOAL_CHIPS" in script.text
        assert rig.client.get("/static/ui.js").status_code == 200
