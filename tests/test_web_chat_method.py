"""The conversation page's search method (step H5, after the first hosted test): look before
you move, say why, pivot from evidence, keep a best-so-far and stop, end honestly, and write
each conversation down without images."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from miroir.config import Settings
from miroir.jobs.openrouter import JSON
from miroir.web import chat as chat_module
from miroir.web.chat import BRIEF, LOCKED_NOTE
from miroir.web.chat_tools import NOT_FOUND, ToolName
from miroir.web.chatlog import FileChatLog, pull, redact
from miroir.web.toolsearch import SearchKind, ToolQuery, ToolSearch
from tests import web_helpers as h
from tests.test_web_chat import (
    EXACT,
    LOOK,
    PHOTO,
    WHY,
    Rig,
    body_text,
    call,
    card_ids,
    of,
    searching,
    shown,
    tool_messages,
)

NOT_SURE = 1.5
"""A cut-off no cosine reaches: the first look rates nothing "Très proche"."""

ASK: dict[str, JSON] = {"question": "Où est-ce ?", "choices": ["En ville", "À la campagne"]}


def last_tools(rig: Rig) -> list[str]:
    return tool_messages(rig.model.bodies[-1])


# --- 1. look before you move -------------------------------------------------------------------


def test_no_search_runs_before_the_first_looks_top_results_are_viewed() -> None:
    script = [
        searching("image_search", {"model": "A3"}),
        LOOK,
        searching("image_search", {"model": "A3"}),
        shown("OBJ1"),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
    refused, looked, searched = last_tools(rig)
    assert refused.startswith("not run: look before you search")
    assert "OBJ1, PA1, OBJ2" in refused
    assert looked.startswith("view_results")
    assert searched.startswith("image_search")
    kinds = [q.kind for q in rig.backend.queries]
    assert kinds == [SearchKind.VIEW, SearchKind.IMAGE]
    assert "Look before you search" in body_text(rig.model.bodies[0])


def test_words_alone_need_no_look_first() -> None:
    with Rig([searching("text_search", {"query": "église"}), shown("PA1")]) as rig:
        cid = rig.start(photo=False, words="église")
        rig.send(cid, "goal", EXACT)
    assert not last_tools(rig)[0].startswith("not run")


# --- 2. say why --------------------------------------------------------------------------------


def test_a_search_or_look_without_why_is_refused_with_an_error_the_model_can_fix() -> None:
    script = [
        call("view_results", {"entity_ids": ["OBJ1"]}),
        call("view_results", {"entity_ids": ["OBJ1"], "why": "   "}),
        LOOK,
        call("text_search", {"query": "église"}),
        shown("PA1"),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        events = rig.send(cid, "goal", EXACT)
    replies = last_tools(rig)
    assert replies[0].startswith("error: view_results needs why")
    assert replies[1].startswith("error: view_results needs why")
    assert replies[3].startswith("error: text_search needs why")
    assert "call again with it" in replies[3]
    assert [q.kind for q in rig.backend.queries] == [SearchKind.VIEW]
    done = [s for s in of(events, "step") if s["state"] == "done" and s.get("why")]
    assert [s["why"] for s in done] == [WHY]


def test_the_why_rides_on_the_running_and_the_done_step() -> None:
    script = [LOOK, searching("image_search", {"model": "auto"}), shown("OBJ1")]
    with Rig(script) as rig:
        cid = rig.start()
        events = rig.send(cid, "goal", EXACT)
        page = rig.client.get("/static/chat.js").text
    searches = [s for s in of(events, "step") if "Recherche par image" in str(s["label"])]
    assert {str(s["state"]) for s in searches} == {"running", "done"}
    assert all(s["why"] == WHY for s in searches)
    assert "Pourquoi : " in page


# --- 3. pivot from evidence --------------------------------------------------------------------


def test_more_like_starts_from_the_stored_vectors_of_a_record_and_leaves_it_out() -> None:
    engine, encoders = h.engine()
    row = engine.catalog.image_row["j1"]
    a7, a3 = engine.stored_cosines(row)
    np.testing.assert_allclose(a7, encoders.matrix @ encoders.matrix[row], rtol=1e-5)
    np.testing.assert_allclose(a3, a7)
    found = ToolSearch(engine).run(
        ToolQuery(SearchKind.LIKE, entity_ids=("OBJ1",), image_id="j2", a7_weight=1.0)
    )
    ids = [x.entity_id for x in found.hits]
    assert "OBJ1" not in ids
    assert set(ids) == {"PA1", "OBJ2"}
    by_j2 = encoders.matrix @ encoders.matrix[engine.catalog.image_row["j2"]]
    best = {"PA1": max(by_j2[0], by_j2[1], by_j2[5]), "OBJ2": by_j2[4]}
    assert ids == sorted(best, key=lambda e: -best[e])


def test_more_like_without_a_seen_image_starts_from_the_cover() -> None:
    engine, _ = h.engine()
    tools = ToolSearch(engine)
    pa1 = engine.catalog.entity_row["PA1"]
    assert tools.start_row(pa1, None) == engine.catalog.image_row["m1"]
    assert tools.start_row(pa1, "j1") == engine.catalog.image_row["m1"]  # not PA1's image
    assert tools.start_row(pa1, "m2") == engine.catalog.image_row["m2"]


def test_more_like_in_the_chat_uses_the_image_the_model_looked_at() -> None:
    script = [
        LOOK,
        searching("more_like", {"entity_id": "PA999"}),
        searching("more_like", {"entity_id": "OBJ1", "model": "A7"}),
        shown("PA1"),
    ]
    with Rig(script) as rig:
        cid = rig.start()
        events = rig.send(cid, "goal", EXACT)
    _, unseen, ran = last_tools(rig)
    assert "PA999 never came from a search" in unseen
    assert ran.startswith("more_like OBJ1 (from the image you looked at")
    like = [q for q in rig.backend.queries if q.kind is SearchKind.LIKE]
    assert len(like) == 1
    assert like[0].entity_ids == ("OBJ1",)
    assert like[0].image_id in {"j1", "j2"}
    assert like[0].a7_weight == 1.0
    labels = [str(s["label"]) for s in of(events, "step") if s["state"] == "done"]
    assert any(label.startswith("Ce qui ressemble à") for label in labels)


def test_a_long_text_query_is_refused() -> None:
    long = "plat or gallo-romain médaillons Bacchus Apollon Diane divini"
    script = [searching("text_search", {"query": long}), shown("PA1", found="none")]
    with Rig(script) as rig:
        cid = rig.start(photo=False, words="plat en or")
        rig.send(cid, "goal", EXACT)
    assert "a text query holds 2 to 5 words" in last_tools(rig)[0]
    assert rig.backend.queries == []


def test_the_brief_asks_for_words_from_evidence_never_guesses() -> None:
    assert "2 to 5 words taken from the visitor's words" in BRIEF
    assert "Never search a name, a place, a god, a maker or an attribution you only" in BRIEF
    assert ToolName.MORE_LIKE.value in BRIEF


# --- 4. keep a best-so-far and stop ------------------------------------------------------------


def test_three_searches_without_a_sure_match_leave_only_a_question_or_the_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_module, "MAX_STALE", 99)  # this test is about the count alone
    script = [
        LOOK,
        searching("image_search", {"model": "A3"}),
        searching("text_search", {"query": "église portail"}),
        searching("more_like", {"entity_id": "PA1"}),
        searching("image_search", {"model": "A7"}),
        searching("view_results", {"entity_ids": ["OBJ2"]}),
        call("ask_visitor", ASK),
    ]
    with Rig(script, cutoff=NOT_SURE) as rig:
        cid = rig.start()
        events = rig.send(cid, "goal", EXACT)
        assert events[-1]["waiting"] == "answer"
        assert [q.kind for q in rig.backend.queries].count(SearchKind.LIKE) == 1
        assert len(rig.backend.queries) == 4  # the look and three searches
        replies = last_tools(rig)
        assert replies[4].startswith("not run: 3 searches without a sure match")
        assert replies[5].startswith("not run: 3 searches")  # no more looks either
        assert rig.model.bodies[4]["tool_choice"] == "required"
        assert LOCKED_NOTE.split("{", 1)[0] in body_text(rig.model.bodies[4])
        # The visitor's answer starts a new message: searching works again.
        rig.model._answers += [searching("image_search", {"model": "mix"}), shown("PA1")]
        rig.send(cid, "answer", "En ville")
    assert not last_tools(rig)[-2].startswith("not run")


def test_two_searches_in_a_row_that_add_nothing_new_lock_the_turn() -> None:
    script = [
        LOOK,  # the first look's three entities are already the best so far
        searching("image_search", {"model": "A3"}),
        searching("image_search", {"model": "A7"}),
        searching("text_search", {"query": "église"}),
        shown("OBJ1"),
    ]
    with Rig(script) as rig:  # a sure first look: the count of 3 does not apply
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
    replies = last_tools(rig)
    assert replies[3].startswith("not run: 2 searches in a row brought nothing new")
    assert "Best so far: OBJ1" in replies[1]


def test_a_search_that_brings_a_new_best_resets_the_stale_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_module, "MAX_SEARCHES", 99)
    script = [
        searching("text_search", {"query": "église"}),
        searching("text_search", {"query": "vase", "best": ["PA1"]}),  # PA1 came from the 1st
        searching("text_search", {"query": "nature morte"}),
        searching("text_search", {"query": "lièvre"}),
        shown("PA1"),
    ]
    with Rig(script) as rig:
        cid = rig.start(photo=False, words="église")
        rig.send(cid, "goal", EXACT)
    replies = last_tools(rig)
    assert [r.startswith("not run") for r in replies[:4]] == [False, False, False, True]
    assert "Best so far: PA1 (Église Saint-Pierre" in replies[1]


def test_best_names_only_records_the_model_has_seen() -> None:
    script = [
        searching("text_search", {"query": "église", "best": ["PA999"]}),
        searching("view_results", {"entity_ids": ["PA1"], "best": ["PA999", "PA1"]}),
        shown("PA1"),
    ]
    with Rig(script) as rig:
        cid = rig.start(photo=False, words="église")
        rig.send(cid, "goal", EXACT)
    replies = last_tools(rig)
    assert "Best so far: none named yet" in replies[0]
    assert "Best so far: PA1 (" in replies[1]
    assert "PA999" not in replies[1]


# --- 5. end honestly ---------------------------------------------------------------------------


def test_show_results_must_say_what_it_found() -> None:
    script = [
        call("show_results", {"results": [{"id": "PA1", "reason": "r"}]}),
        call("show_results", {"found": "close", "results": [{"id": "PA1", "reason": "r"}]}),
        call("show_results", {"found": "exact", "results": [{"id": "PA1", "reason": "r"}]}),
        shown("PA1"),
    ]
    with Rig(script, cutoff=NOT_SURE) as rig:
        cid = rig.start()
        events = rig.send(cid, "goal", EXACT)
    no_found, no_summary, unchecked = last_tools(rig)
    assert "show_results needs found: exact" in no_found
    assert "with found close, write summary" in no_summary
    assert "found exact needs a first pick you have checked by eye" in unchecked
    assert of(events, "results")[0]["found"] == "close"
    assert "Je ne trouve pas cet objet précis" in str(of(events, "say")[-1]["text"])


def test_exact_needs_a_look_or_the_sites_very_close() -> None:
    looked = [LOOK, shown("PA1", found="exact")]
    with Rig(looked, cutoff=NOT_SURE) as rig:
        cid = rig.start()
        events = rig.send(cid, "goal", EXACT)
    assert card_ids(events) == ["PA1"]
    assert of(events, "results")[0]["found"] == "exact"
    with Rig([shown("OBJ1", found="exact")]) as rig:  # OBJ1 rated "Très proche"
        cid = rig.start()
        assert card_ids(rig.send(cid, "goal", EXACT)) == ["OBJ1"]


def test_the_summary_is_said_once_not_twice() -> None:
    reply = shown("PA1")
    script = [
        searching("text_search", {"query": "église"}),
        replace(reply, text="Je n’ai pas trouvé ce plat précis."),
    ]
    with Rig(script) as rig:
        cid = rig.start(photo=False, words="plat")
        events = rig.send(cid, "goal", EXACT)
    assert [e["text"] for e in of(events, "say")] == [
        "Je ne trouve pas cet objet précis. Les plus proches."
    ]


def test_found_none_says_so_plainly_even_when_the_model_did_not() -> None:
    script = [call("show_results", {"found": "none", "summary": "Rien de tel ici.", "results": []})]
    with Rig(script) as rig:
        cid = rig.start(photo=False, words="patère de Rennes")
        events = rig.send(cid, "goal", EXACT)
    said = str(of(events, "say")[-1]["text"])
    assert said == f"{NOT_FOUND} Rien de tel ici."
    assert of(events, "results")[0]["cards"] == []
    assert of(events, "results")[0]["found"] == "none"


# --- 6. the goal chips -------------------------------------------------------------------------


def test_the_goal_chips_ask_what_where_more_and_who() -> None:
    with Rig([]) as rig:
        page = rig.client.get("/static/chat.js").text
    for label in ("C’est quoi ?", "C’est où ?", "D’autres comme ça", "Qui l’a fait ?"):
        assert f"label: '{label}'" in page
    assert "Cet objet précis" not in page


# --- 7. the log --------------------------------------------------------------------------------


class Commits:
    def __init__(self) -> None:
        self.count = 0

    async def __call__(self) -> None:
        self.count += 1


def test_each_conversation_is_logged_without_images(tmp_path: Path) -> None:
    commits = Commits()
    log = FileChatLog(tmp_path, commits)
    near: dict[str, JSON] = {"near": {"lat": 43.60441, "lon": 1.44295, "km": 2}}
    script = [
        LOOK,
        searching("image_search", {"model": "A3", "filters": near}),
        shown("PA1"),
    ]
    with Rig(script, log=log) as rig:
        cid = rig.start(words="un portail")
        rig.send(cid, "goal", EXACT)
    files = list(tmp_path.glob("2026-10-16/*.jsonl"))
    assert [f.stem for f in files] == [cid]
    text = files[0].read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines()]
    kinds = [line["type"] for line in lines]
    assert kinds[:3] == ["open", "visitor", "first_look"]
    assert kinds[-1] == "end"
    assert lines[0]["words"] == "un portail"
    assert lines[1]["text"] == EXACT
    assert set(lines[2]["hits"]) == {"OBJ1", "PA1", "OBJ2"}
    tools = [line for line in lines if line["type"] == "tool"]
    assert [t["name"] for t in tools] == ["view_results", "image_search", "show_results"]
    assert tools[1]["why"] == WHY
    assert tools[1]["hits"]
    assert tools[1]["args"]["filters"]["near"] == "(position)"
    assert tools[2]["found"] == "close"
    assert tools[2]["picks"] == ["PA1"]
    models = [line for line in lines if line["type"] == "model"]
    assert len(models) == 3
    assert models[0]["cost_usd"] == pytest.approx(0.001)
    assert "ms" in models[0]
    assert lines[-1]["spent_usd"] == pytest.approx(0.003)
    for banned in ("data:image", "base64", PHOTO[:40], "43.60441"):
        assert banned not in text
    assert commits.count == 1


def test_a_refused_call_is_logged_with_its_reason(tmp_path: Path) -> None:
    script = [searching("image_search", {"model": "A3"}), LOOK, shown("OBJ1")]
    with Rig(script, log=FileChatLog(tmp_path)) as rig:
        cid = rig.start()
        rig.send(cid, "goal", EXACT)
    lines = [
        json.loads(x)
        for x in (tmp_path / "2026-10-16" / f"{cid}.jsonl").read_text().split("\n")
        if x
    ]
    refused = [x for x in lines if x["type"] == "tool" and "refused" in x]
    assert refused[0]["refused"].startswith("not run: look before you search")


def test_a_log_with_an_unsafe_name_is_skipped(tmp_path: Path) -> None:
    log = FileChatLog(tmp_path)
    log.write("2026-10-16", "../escape", {"type": "open"})
    assert list(tmp_path.rglob("*")) == []
    asyncio.run(log.flush())  # no commit given: nothing to do


def test_redact_hides_a_near_point_and_keeps_the_rest() -> None:
    args = {"why": "w", "filters": {"near": {"lat": 1.0, "lon": 2.0}, "commune": ["Rennes"]}}
    assert redact(args) == {"why": "w", "filters": {"near": "(position)", "commune": ["Rennes"]}}
    assert redact({"filters": {"near": "visitor"}}) == {"filters": {"near": "visitor"}}


def test_pulled_logs_keep_their_layout_and_skip_what_is_already_here(tmp_path: Path) -> None:
    store = {
        "2026-10-16/abc.jsonl": b'{"type": "open"}\n',
        "2026-10-16/../x.jsonl": b"no",
        "loose.jsonl": b"no",
        "2026-10-17/def.jsonl": b'{"type": "end"}\n',
    }
    first = pull(lambda: list(store), store.__getitem__, tmp_path)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in first) == [
        "2026-10-16/abc.jsonl",
        "2026-10-17/def.jsonl",
    ]
    assert pull(lambda: list(store), store.__getitem__, tmp_path) == []
    store["2026-10-16/abc.jsonl"] += b'{"type": "end"}\n'
    again = pull(lambda: list(store), store.__getitem__, tmp_path)
    assert [p.name for p in again] == ["abc.jsonl"]


def test_local_logs_live_in_the_data_dir(settings: Settings) -> None:
    assert settings.chat_logs_dir == settings.data_root / "chat-logs"
