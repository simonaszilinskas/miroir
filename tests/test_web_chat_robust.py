"""The agentic search (step H5) when things go wrong: a search that fails, a model that is slow
or down, a stream that drops, calls in flight against the spending caps."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence

import httpx
import pytest
from pydantic import SecretStr

from miroir.jobs.mistral import PRICES
from miroir.jobs.openrouter import JSON, ChatAnswer
from miroir.web import chat as chat_module
from miroir.web import chat_model, chat_prompts
from miroir.web.chat import (
    ChatModel,
    ChatService,
    Conversation,
    Limits,
    MemorySpend,
    ModelError,
    OpenRouterModel,
    SpendStore,
    ThumbSource,
    Visit,
    VisitorInput,
)
from miroir.web.chatlog import ChatLog
from miroir.web.fake import ScriptedModel
from miroir.web.publishable import Publishable
from tests.test_web_chat import (
    CAPS,
    EXACT,
    WHY,
    CountingBackend,
    Fetches,
    Rig,
    body_text,
    calls,
    card_ids,
    of,
    shown,
    tool_messages,
)


def make_service(
    answers: Sequence[ChatAnswer],
    *,
    log: ChatLog | None = None,
    model: ChatModel | None = None,
    spend: SpendStore | None = None,
    limits: Limits = CAPS,
) -> ChatService:
    """The conversation service alone, without the site, to drive it from one event loop."""
    backend = CountingBackend()
    return ChatService(
        backend=backend,
        model=model or ScriptedModel(answers),
        thumbs=ThumbSource(Publishable(["m1", "j1", "r1"]), Fetches()),
        spend=spend or MemorySpend(),
        limits=limits,
        index_size=(6, 3),
        catalog=backend.engine.catalog,
        today=lambda: "2026-10-16",
        log=log,
    )


def assert_valid_history(bodies: Sequence[Mapping[str, JSON]]) -> None:
    """Every tool call the model made has its reply, right after it, in every request."""
    for body in bodies:
        messages = body["messages"]
        assert isinstance(messages, list)
        for i, message in enumerate(messages):
            assert isinstance(message, dict)
            asked = message.get("tool_calls")
            if not isinstance(asked, list):
                continue
            ids = [str(c["id"]) for c in asked if isinstance(c, dict)]
            replies = messages[i + 1 : i + 1 + len(ids)]
            got = [str(r["tool_call_id"]) for r in replies if isinstance(r, dict)]
            assert got == ids, f"message {i}: calls {ids}, replies {got}"


# --- 1. a tool that raises ---------------------------------------------------------------------


def test_a_search_that_fails_gets_an_error_reply_and_the_turn_goes_on() -> None:
    """A transport error from the search backend once left tool calls without replies: every
    later model call was refused, and the conversation was dead."""
    script = [
        calls(
            [
                ("text_search", {"query": "église", "why": WHY}),
                ("text_search", {"query": "vase céramique", "why": WHY}),
            ]
        ),
        shown("OBJ1"),
    ]
    with Rig(script) as rig:
        rig.backend.failures.append(RuntimeError("connection reset"))
        cid = rig.start(photo=False, words="église")
        events = rig.send(cid, "goal", EXACT)
    assert not of(events, "error")
    failed, second = tool_messages(rig.model.bodies[1])
    assert failed.startswith("error: text_search failed (RuntimeError)")
    assert second.startswith("text_search 'vase céramique'")
    assert card_ids(events) == ["OBJ1"]
    assert_valid_history(rig.model.bodies)
    assert any(s.get("error") for s in of(events, "step"))


def test_a_search_that_failed_may_be_tried_again() -> None:
    """A failed search is not a repeat: its signature is noted only once it ran."""
    search = ("text_search", {"query": "église", "why": WHY})
    script = [calls([search]), calls([search]), shown("PA1")]
    with Rig(script) as rig:
        rig.backend.failures.append(RuntimeError("connection reset"))
        cid = rig.start(photo=False, words="église")
        rig.send(cid, "goal", EXACT)
    failed, again = tool_messages(rig.model.bodies[2])
    assert failed.startswith("error: text_search failed (RuntimeError)")
    assert again.startswith("text_search 'église'")


# --- 2. a first look that fails ----------------------------------------------------------------


def test_a_first_look_that_fails_still_gives_the_model_the_photo_and_the_goal() -> None:
    script = [calls([("text_search", {"query": "église", "why": WHY})]), shown("PA1")]
    with Rig(script) as rig:
        rig.backend.search_failure = ConnectionError("engine down")
        cid = rig.start()
        events = rig.send(cid, "goal", EXACT)
    assert not of(events, "error")
    assert card_ids(events) == ["PA1"]
    opening = body_text(rig.model.bodies[0])
    assert EXACT in opening
    assert "data:image/jpeg;base64," in opening
    assert "the usual search did not answer" in opening
    assert any(s.get("error") for s in of(events, "step"))


def test_an_opening_that_fails_is_built_again_with_the_next_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conversation counts as started only once the model has its opening."""
    with Rig([shown("PA1")]) as rig:
        assert rig.service is not None
        service = rig.service
        real = service._opening
        failures = [KeyError("boom")]

        async def opening(c: Conversation, goal: str) -> dict[str, JSON]:
            if failures:
                raise failures.pop()
            return await real(c, goal)

        monkeypatch.setattr(service, "_opening", opening)
        cid = rig.start()
        first = rig.send(cid, "goal", EXACT)
        assert of(first, "error")
        second = rig.send(cid, "goal", EXACT)
    assert card_ids(second) == ["PA1"]
    messages = rig.model.bodies[0]["messages"]
    assert isinstance(messages, list)
    assert [m["role"] for m in messages if isinstance(m, dict)] == ["system", "user"]
    assert EXACT in body_text(rig.model.bodies[0])


# --- 3. the next message right after a turn ----------------------------------------------------


class SlowLog:
    """A log whose flush takes a while, as a commit to a volume does."""

    def __init__(self) -> None:
        self.flushes = 0

    def write(self, day: str, conversation: str, line: Mapping[str, JSON]) -> None:
        del day, conversation, line

    async def flush(self) -> None:
        await asyncio.sleep(0.5)
        self.flushes += 1


def test_the_visitor_may_write_again_as_soon_as_the_turn_ends() -> None:
    """The turn once awaited the log's flush after its end event, still running: a quick
    next message got 409 "Le modèle travaille encore"."""
    script = [
        calls([("text_search", {"query": "église", "why": WHY})]),
        shown("PA1"),
        calls([("text_search", {"query": "vase", "why": WHY})]),
        shown("OBJ1"),
    ]
    log = SlowLog()
    service = make_service(script, log=log)

    async def run() -> list[str]:
        c = service.open("s", photo=None, words="église", position=None)
        ends: list[str] = []
        for kind, text in ((Visit.GOAL, EXACT), (Visit.MESSAGE, "plutôt un vase")):
            after = service.send(c, VisitorInput(kind, text))
            events = [e async for e in service.stream(c, after)]
            ends.append(str(events[-1]["type"]))
        await asyncio.sleep(0.6)
        return ends

    started = time.monotonic()
    assert asyncio.run(run()) == ["end", "end"]
    assert time.monotonic() - started < 3  # each stream closed at its end, not at a ping
    assert log.flushes == 2


# --- 4. a model that is slow or down -----------------------------------------------------------


class Wire:
    """OpenRouter's answers, in order: an HTTP status with a body, or an error to raise."""

    def __init__(self, answers: Sequence[httpx.Response | Exception]) -> None:
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        del request
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def chat_body(cost: float | None = 0.002) -> httpx.Response:
    usage: dict[str, JSON] = {"prompt_tokens": 1000, "completion_tokens": 100}
    if cost is not None:
        usage["cost"] = cost
    return httpx.Response(
        200,
        json={
            "id": "gen-1",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": usage,
        },
    )


def complete(wire: Wire) -> ChatAnswer:
    model = OpenRouterModel(SecretStr("sk-or-test"), transport=httpx.MockTransport(wire))
    return asyncio.run(model.complete({"messages": [], "max_tokens": 800}))


@pytest.fixture
def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_model, "RETRY_WAIT_S", 0.0)


@pytest.mark.usefixtures("no_wait")
def test_a_model_call_that_timed_out_is_not_sent_again() -> None:
    """It may have run and been paid for: sending it again could pay up to six times."""
    wire = Wire([httpx.ReadTimeout("slow"), chat_body()])
    with pytest.raises(ModelError) as raised:
        complete(wire)
    assert raised.value.maybe_billed
    assert wire.calls == 1


@pytest.mark.usefixtures("no_wait")
def test_a_busy_model_is_asked_twice_at_most() -> None:
    wire = Wire([httpx.Response(503), httpx.Response(503), chat_body()])
    with pytest.raises(ModelError) as raised:
        complete(wire)
    assert not raised.value.maybe_billed
    assert wire.calls == 2
    assert complete(Wire([httpx.ConnectError("refused"), chat_body()])).text == "ok"


@pytest.mark.usefixtures("no_wait")
def test_a_refused_request_is_not_sent_again() -> None:
    wire = Wire([httpx.Response(400, json={"error": {"message": "bad"}}), chat_body()])
    with pytest.raises(ModelError):
        complete(wire)
    assert wire.calls == 1


def test_an_answer_without_a_cost_is_priced_from_its_tokens() -> None:
    """OpenRouter once answered 200 without a cost: the call raised, and went uncounted."""
    prompt, _, completion = PRICES[chat_model.MISTRAL_MODEL]
    answer = complete(Wire([chat_body(cost=None)]))
    assert answer.cost_usd == pytest.approx((1000 * prompt + 100 * completion) / 1e6)
    assert complete(Wire([chat_body(cost=0.002)])).cost_usd == 0.002


class Raising:
    """A model that sleeps `sleep_s`, then fails with `error` or shows PA1."""

    def __init__(self, error: Exception | None = None, sleep_s: float = 0.0) -> None:
        self.error = error
        self.sleep_s = sleep_s

    @property
    def model(self) -> str:
        return "raising"

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        del body
        await asyncio.sleep(self.sleep_s)
        if self.error is not None:
            raise self.error
        return shown("PA1")


def turn(service: ChatService) -> list[dict[str, JSON]]:
    async def run() -> list[dict[str, JSON]]:
        c = service.open("s", photo=None, words="église", position=None)
        after = service.send(c, VisitorInput(Visit.GOAL, EXACT))
        return [e async for e in service.stream(c, after) if e["type"] != "ping"]

    return asyncio.run(run())


def test_a_model_down_says_so_and_counts_a_call_that_may_have_been_paid() -> None:
    spend = MemorySpend()
    down = Raising(ModelError("no answer", maybe_billed=True))
    events = turn(make_service([], model=down, spend=spend))
    assert [e["text"] for e in of(events, "error")] == [chat_prompts.MODEL_DOWN]
    assert asyncio.run(spend.spent("2026-10-16")) == pytest.approx(chat_module.FIRST_CALL_USD)


def test_a_turn_stops_at_its_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_module, "TURN_S", 0.2)
    started = time.monotonic()
    events = turn(make_service([], model=Raising(sleep_s=30)))
    assert time.monotonic() - started < 5
    assert [e["text"] for e in of(events, "error")] == [chat_prompts.OUT_OF_TIME]
    assert events[-1]["type"] == "end"
