"""Step H4: a vision model leads the search (experiments.md H4, technical spec §7.4).

The model sees the query photo and drives the tools of `miroir.eval.h4_tools`. Three ways:

- S1, one-shot plan: one call returns a JSON plan (image model weights, a text query and its
  weight, filters); code carries it out. The model never sees a result.
- S2, plan and review: S1, then one more call shows the model the plan's top 20 as 256 px
  thumbnails with their titles; it puts up to 10 of them in order.
- S3, free loop: the model calls tools as it likes, up to `MAX_TOOL_CALLS`, then must call
  `final_answer`. If it never does, its last result list is the answer.

H1 is redone with the same model as a baseline: it only describes the photo (`describe.PROMPT`).

Rules, as for every paid run: only publishable images (and Commons photos) go out, checked
here on every image id; every reply is saved as it arrives, keyed by its request, so a rerun
replays it at no cost (`CallCache`); every cent goes to the project ledger, and a run stops at
its `limit_usd`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter

from miroir.commons.images import CommonsImages
from miroir.eval.h4_tools import (
    BOOST,
    MAX_ANSWER,
    MAX_TOP_K,
    MAX_VIEW,
    Base,
    FilterMode,
    Filters,
    QueryData,
    Session,
    Tool,
    World,
    answer_scores,
    parse_plan,
    parse_ranking,
    plan_scores,
    top,
)
from miroir.eval.scoring import Floats
from miroir.images.fetch import LONG_SIDE, ImageFetcher, to_jpeg
from miroir.jobs.describe import PROMPT as H1_PROMPT
from miroir.jobs.openrouter import JSON, ChatAnswer, OpenRouterError, ToolCall
from miroir.jobs.privacy import PublishableImages
from miroir.jobs.spend import BudgetError, SpendKind, SpendLedger

logger = logging.getLogger(__name__)

#: Tool calls S3 may make before it must call final_answer (final_answer not counted).
MAX_TOOL_CALLS = 8
#: Model calls S3 may make in all: tool turns, reminders, the last forced turn.
MAX_TURNS = 12
#: Results S2 shows the model.
REVIEW_SIZE = 20
#: Long side of the thumbnails the model sees.
THUMB_PX = 256
#: Requests that fail in a row before the run stops (out of credit, model gone).
MAX_FAILURES_IN_A_ROW = 5
MAX_TOKENS = {"H1": 700, "S1": 700, "S2": 400, "S3": 800}


class Scenario(StrEnum):
    H1 = "H1"
    """Baseline: the model describes the photo; its words are searched (and mixed with A7)."""
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


@dataclass(frozen=True, slots=True)
class AgentOption:
    option: str
    """Ledger label, e.g. "H4-mistral-large-4"."""
    model: str
    """OpenRouter model id."""
    slug: str
    extra: Mapping[str, JSON]
    """Request fields beyond messages, tools and max_tokens."""


OPTIONS: dict[str, AgentOption] = {
    "mistral-large-4": AgentOption(
        "H4-mistral-large-4",
        "mistralai/mistral-large-4-0",
        "mistral-large-4",
        # Reasoning is on by default ("high"); off keeps replies short and the cost known.
        extra={"temperature": 0.0, "reasoning": {"effort": "none"}},
    ),
}


# --- images ----------------------------------------------------------------------------------


class ImageSource:
    """JPEG bytes of the images the model may see: publishable POP images and Commons photos.

    Every id is checked with `PublishableImages` before its bytes are read, so a private
    image cannot reach the model even if a caller asks for one.
    """

    def __init__(
        self,
        store: PublishableImages,
        fetcher: ImageFetcher,
        commons: CommonsImages,
        service: str,
    ) -> None:
        self._store = store
        self._fetcher = fetcher
        self._commons = commons
        self._service = service
        self._lock = threading.Lock()
        self._memo: dict[tuple[str, int], bytes] = {}

    def check(self, image_ids: Sequence[str]) -> None:
        self._store.check(image_ids, self._service)

    def jpeg(self, image_id: str, px: int) -> bytes:
        self.check([image_id])
        with self._lock:
            known = self._memo.get((image_id, px))
        if known is not None:
            return known
        if image_id in self._store.images:
            fetched = self._fetcher.fetch(self._store.images[image_id])
        else:
            with self._lock:  # the Commons client keeps Wikimedia's pace, one request at a time
                fetched = self._commons.fetch(image_id)
        if fetched.jpeg is None:
            msg = f"{image_id}: could not fetch the image ({fetched.skip})"
            raise OpenRouterError(msg)
        data = fetched.jpeg if px >= LONG_SIDE else to_jpeg(fetched.jpeg, px)
        with self._lock:
            self._memo[(image_id, px)] = data
        return data


# --- calls, saved and replayed ---------------------------------------------------------------


class ChatClient(Protocol):
    def chat(self, model: str, body: Mapping[str, JSON]) -> ChatAnswer: ...


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One model reply, as saved. `cost_usd` is what it cost when it was paid for."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    key: str
    model: str
    query_id: str
    scenario: str
    step: str
    text: str
    tool_calls: tuple[ToolCall, ...]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float
    finish_reason: str
    at: datetime


_RECORD = TypeAdapter(CallRecord)


class CallCache:
    """Replies by request key, in a JSON-lines file; the last line wins for a key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._by_key: dict[str, CallRecord] = {}
        if path.exists():
            with path.open("rb") as lines:
                for line in lines:
                    if line.strip():
                        record = _RECORD.validate_json(line)
                        self._by_key[record.key] = record

    def get(self, key: str) -> CallRecord | None:
        with self._lock:
            return self._by_key.get(key)

    def put(self, record: CallRecord) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as out:
                out.write(_RECORD.dump_json(record) + b"\n")
                out.flush()
            self._by_key[record.key] = record


@dataclass(frozen=True, slots=True)
class Reply:
    record: CallRecord
    replayed: bool
    """True when it came from the cache: nothing was paid this time."""

    @property
    def text(self) -> str:
        return self.record.text

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.record.tool_calls


type Message = dict[str, JSON]


def text_part(text: str) -> JSON:
    return {"type": "text", "text": text}


def image_part(image_id: str, px: int) -> JSON:
    """An image by id. It becomes a data URL only when sent, so keys never hold image bytes."""
    return {"type": "image_ref", "image_id": image_id, "px": px}


def request_key(model: str, body: Mapping[str, JSON]) -> str:
    """Same model and request (images by id and size) give the same key."""
    canonical = json.dumps({"model": model, "body": body}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Caller:
    """Sends requests for one model, replays saved replies, and keeps spend under a limit."""

    def __init__(
        self,
        option: AgentOption,
        client: ChatClient,
        images: ImageSource,
        cache: CallCache,
        ledger: SpendLedger,
        *,
        limit_usd: float,
    ) -> None:
        self.option = option
        self._client = client
        self._images = images
        self._cache = cache
        self._ledger = ledger
        self.limit_usd = limit_usd
        self._lock = threading.Lock()
        self.spent_usd = 0.0
        self._failures = 0

    def check_images(self, image_ids: Sequence[str]) -> None:
        """Fail before paying if any image may not go out."""
        self._images.check(image_ids)

    def ask(
        self,
        messages: Sequence[Message],
        *,
        query_id: str,
        scenario: Scenario,
        step: str,
        tools: Sequence[JSON] | None = None,
        tool_choice: JSON | None = None,
        json_mode: bool = False,
    ) -> Reply:
        body: dict[str, JSON] = {
            "messages": list(messages),
            "max_tokens": MAX_TOKENS[scenario.value],
            **self.option.extra,
        }
        if tools is not None:
            body["tools"] = list(tools)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        key = request_key(self.option.model, body)
        saved = self._cache.get(key)
        if saved is not None:
            return Reply(saved, replayed=True)
        with self._lock:
            if self.spent_usd >= self.limit_usd:
                msg = f"{self.option.option}: spent ${self.spent_usd:.4f}, at its limit"
                raise BudgetError(msg)
        sent = {**body, "messages": [self._render(m) for m in messages]}
        try:
            answer = self._client.chat(self.option.model, sent)
        except OpenRouterError:
            with self._lock:
                self._failures += 1
                if self._failures >= MAX_FAILURES_IN_A_ROW:
                    msg = f"{self._failures} requests failed in a row; stopped"
                    raise BudgetError(msg) from None
            raise
        record = CallRecord(
            key=key,
            model=self.option.model,
            query_id=query_id,
            scenario=scenario.value,
            step=step,
            text=answer.text,
            tool_calls=answer.tool_calls,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            cached_tokens=answer.cached_tokens,
            cost_usd=answer.cost_usd,
            finish_reason=answer.finish_reason,
            at=datetime.now(UTC),
        )
        self._ledger.record(
            SpendKind.RUN,
            option=self.option.option,
            run=f"{self.option.option}-{scenario.value}",
            model=self.option.model,
            cost_usd=answer.cost_usd,
            items=1,
        )
        self._cache.put(record)
        with self._lock:
            self._failures = 0
            self.spent_usd += answer.cost_usd
        return Reply(record, replayed=False)

    def _render(self, message: Message) -> JSON:
        content = message.get("content")
        if not isinstance(content, list):
            return message
        parts: list[JSON] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_ref":
                image_id = str(part["image_id"])
                px = int(str(part["px"]))
                data = base64.b64encode(self._images.jpeg(image_id, px)).decode()
                url = f"data:image/jpeg;base64,{data}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                parts.append(part)
        return {**message, "content": parts}


# --- prompts ---------------------------------------------------------------------------------

_BASES = ", ".join(b.value for b in Base)

CONTEXT = f"""You help find French heritage records (the POP database) that match a photo.
The index holds about 5,800 buildings and objects. Each has an id and one or more images:
- merimee: buildings (ids PA...). Their images are archive prints from the Médiathèque du \
patrimoine (mostly black and white, often glued on a card with a typed caption naming the \
commune and the building), plus some recent colour photos from Wikimedia Commons.
- palissy: objects kept in buildings (ids PM...), photographed in the same archive.
- joconde: museum objects, often several views of one object.
- rose-valland: works looted during the war (MNR) and Jeu de Paume room photos.
- memoire: archive photos tied to no building or object.
The photo you get may be an archive print of a building, a visitor's colour photo of a \
building, a museum object, or a black and white crop of an artwork hanging in a room. The \
right answer is the building or object it shows, never the photo itself; other photos taken \
on the same day are hidden, so you must find the building from another visit.

What is known from past tests:
- A7 (EmbeddingGemma 2) finds black and white archive prints of buildings best.
- A3 (MegaLoc) finds colour photos of buildings and museum objects best. A good default: A7 \
alone for a grey building photo, A3 alone for a colour building photo, A7 weight 0.2 for an \
object.
- Text search (BM25 over record titles, descriptions, communes, départements, periods, \
authors) wins when the photo shows readable text, above all a typed caption naming a place: \
copy those words. It is French only: write French words as the records use them \
("église", "portail", "retable", the commune's name).
- Filters keep (hard) or favour (boost, +{BOOST} on a 0-1 score) records whose base \
({_BASES}), département, commune, domain or period matches. A hard filter on a wrong guess \
loses the right answer; boost is safer when you are not sure."""

S1_TASK = """Plan one search for this photo. You will not see the results. Answer with one \
JSON object only:
{
  "reading": "what the photo shows and any text you can read, in a few words",
  "image_search": {"model": "A7" | "A3" | "mix", "a7_weight": 0.0 to 1.0 (for mix)},
  "text_search": {"query": "French words to search, or empty", "weight": 0.0 to 1.0 \
(share of the text score in the mix with the image score; 0 when the query is empty)},
  "filters": {"base": [...], "departement": [...], "commune": [...], "domain": [...], \
"period": [...], "mode": "boost" | "hard"}
}
Leave a filter list empty when you do not know. Use a hard filter only when the photo itself \
tells you (a caption naming the commune, for example)."""

S2_TASK = """Here are the first {n} results of a search for this photo: an id, the base, the \
title and the place of each, then a thumbnail of each result's image closest to the photo. \
Look at the photo and the thumbnails, and put the results that show the same building or \
object as the photo first. Answer with one JSON object only: {{"ranking": [up to 10 ids, best \
first]}}. Use only ids from this list."""

S3_TASK = f"""Find the building or object this photo shows. Call the tools: image_search and \
text_search list results, view_results shows their thumbnails so you can compare them with \
the photo by eye. You may make up to {MAX_TOOL_CALLS} tool calls, then call final_answer \
with up to {MAX_ANSWER} ids, best first. Only ids you have seen in results count."""

S3_LAST_CALL = f"""You have used your {MAX_TOOL_CALLS} tool calls. Call final_answer now with \
up to {MAX_ANSWER} ids, best first."""

S3_NUDGE = "Call a tool, or final_answer to finish."

_FILTERS_SCHEMA: JSON = {
    "type": "object",
    "description": "Favour (boost) or keep only (hard) records matching every field given.",
    "properties": {
        "base": {"type": "array", "items": {"type": "string", "enum": [b.value for b in Base]}},
        "departement": {"type": "array", "items": {"type": "string"}},
        "commune": {"type": "array", "items": {"type": "string"}},
        "domain": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Words of the record's domain or type: église, peinture, monnaie...",
        },
        "period": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string", "enum": [m.value for m in FilterMode]},
    },
}

TOOLS: tuple[JSON, ...] = (
    {
        "type": "function",
        "function": {
            "name": Tool.IMAGE_SEARCH.value,
            "description": "Search by the photo's look: cosine over saved image vectors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "enum": ["A7", "A3", "mix"]},
                    "a7_weight": {
                        "type": "number",
                        "description": "For mix: weight of A7, A3 gets the rest.",
                    },
                    "top_k": {"type": "integer", "description": f"1 to {MAX_TOP_K}, default 10"},
                    "filters": _FILTERS_SCHEMA,
                },
                "required": ["model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": Tool.TEXT_SEARCH.value,
            "description": "BM25 search over record text, in French.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "description": f"1 to {MAX_TOP_K}, default 10"},
                    "filters": _FILTERS_SCHEMA,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": Tool.VIEW_RESULTS.value,
            "description": f"Thumbnails (256 px) of up to {MAX_VIEW} results, with their "
            "titles, to compare with the photo by eye.",
            "parameters": {
                "type": "object",
                "properties": {"entity_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["entity_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": Tool.FINAL_ANSWER.value,
            "description": f"Your answer: up to {MAX_ANSWER} ids, best first. Ends the search.",
            "parameters": {
                "type": "object",
                "properties": {"entity_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["entity_ids"],
            },
        },
    },
)


def _system(text: str) -> Message:
    return {"role": "system", "content": text}


def _user(*parts: JSON) -> Message:
    return {"role": "user", "content": list(parts)}


# --- outcomes ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Step:
    """One tool call of S3, for the trace."""

    turn: int
    tool: str
    arguments: str
    listed: tuple[str, ...]
    filter_mode: str
    """"none", "boost" or "hard" for a search; "" for other tools."""
    error: bool
    run: bool
    """False when it came past the tool budget and was refused."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one scenario did for one query."""

    scenario: Scenario
    query_id: str
    set_id: str
    final: tuple[str, ...]
    """The model's answer, best first (S1: the plan's top 10)."""
    ranking: Floats = field(repr=False)
    """A score per catalog entity: `final` first, then the fallback order."""
    calls: tuple[CallRecord, ...]
    replayed: int
    failure: str = ""
    """Empty, or what went wrong: "no-json", "no-final-answer", "api-error: ..."."""
    detail: Mapping[str, JSON] = field(default_factory=dict[str, JSON])
    steps: tuple[Step, ...] = ()

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def trace(self) -> dict[str, JSON]:
        """One JSON line for the run's trace file."""
        return {
            "scenario": self.scenario.value,
            "query_id": self.query_id,
            "set_id": self.set_id,
            "final": list(self.final),
            "failure": self.failure,
            "calls": len(self.calls),
            "replayed": self.replayed,
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "cached_tokens": sum(c.cached_tokens for c in self.calls),
            "cost_usd": self.cost_usd,
            "detail": dict(self.detail),
            "steps": [
                {
                    "turn": s.turn,
                    "tool": s.tool,
                    "arguments": s.arguments,
                    "listed": list(s.listed[:10]),
                    "filter_mode": s.filter_mode,
                    "error": s.error,
                    "run": s.run,
                }
                for s in self.steps
            ],
        }


def _hidden(world: World, q: QueryData) -> NDArray[np.bool_]:
    return ~np.isfinite(world.g5(q))


def _failed(scenario: Scenario, world: World, q: QueryData, err: Exception) -> Outcome:
    logger.warning("%s %s: %s", scenario.value, q.query.query_id, err)
    return Outcome(
        scenario,
        q.query.query_id,
        q.set_id,
        (),
        world.g5(q),
        (),
        0,
        failure=f"api-error: {str(err)[:200]}",
    )


# --- H1 baseline -----------------------------------------------------------------------------


def run_h1(caller: Caller, world: World, q: QueryData) -> Outcome:
    """The H1 question, unchanged; scoring turns its answer into a text search."""
    messages = [_user(text_part(H1_PROMPT), image_part(q.query.image_id, LONG_SIDE))]
    try:
        reply = caller.ask(
            messages,
            query_id=q.query.query_id,
            scenario=Scenario.H1,
            step="describe",
            json_mode=True,
        )
    except OpenRouterError as err:
        return _failed(Scenario.H1, world, q, err)
    return Outcome(
        Scenario.H1,
        q.query.query_id,
        q.set_id,
        (),
        world.g5(q),
        (reply.record,),
        int(reply.replayed),
        detail={"answer": reply.text},
    )


# --- S1 and S2 -------------------------------------------------------------------------------


def run_s1(caller: Caller, world: World, q: QueryData) -> Outcome:
    messages = [
        _system(CONTEXT),
        _user(text_part(S1_TASK), image_part(q.query.image_id, LONG_SIDE)),
    ]
    try:
        reply = caller.ask(
            messages, query_id=q.query.query_id, scenario=Scenario.S1, step="plan", json_mode=True
        )
    except OpenRouterError as err:
        return _failed(Scenario.S1, world, q, err)
    plan = parse_plan(reply.text)
    _, ranking = plan_scores(world, q, plan)
    final = tuple(e for e, _ in top(ranking, world.catalog, MAX_ANSWER))
    return Outcome(
        Scenario.S1,
        q.query.query_id,
        q.set_id,
        final,
        ranking,
        (reply.record,),
        int(reply.replayed),
        failure="" if plan.valid else "no-json",
        detail={
            "plan": reply.text,
            "parsed": _plan_json(plan.a7_weight, plan.text_query, plan.text_weight, plan.filters),
        },
    )


def _plan_json(a7_weight: float, query: str, text_weight: float, filters: Filters) -> JSON:
    return {
        "a7_weight": a7_weight,
        "text_query": query,
        "text_weight": text_weight,
        "filters": filters.describe(),
        "filter_mode": "none" if filters.empty else filters.mode.value,
    }


def run_s2(caller: Caller, world: World, q: QueryData, s1: Outcome) -> Outcome:
    """Show the model S1's top 20 and let it reorder them; the rest keep S1's order."""
    candidates = [e for e, _ in top(s1.ranking, world.catalog, REVIEW_SIZE)]
    images = world.best_images(q, candidates)
    shown = [e for e in candidates if e in images]
    parts: list[JSON] = [
        text_part(S2_TASK.format(n=len(shown))),
        text_part("The photo:"),
        image_part(q.query.image_id, LONG_SIDE),
    ]
    for i, entity_id in enumerate(shown, start=1):
        parts.append(text_part(f"[{i}] {world.describe(entity_id)}"))
        parts.append(image_part(images[entity_id], THUMB_PX))
    try:
        reply = caller.ask(
            [_system(CONTEXT), _user(*parts)],
            query_id=q.query.query_id,
            scenario=Scenario.S2,
            step="review",
            json_mode=True,
        )
    except OpenRouterError as err:
        failed = _failed(Scenario.S2, world, q, err)
        return Outcome(
            Scenario.S2,
            q.query.query_id,
            q.set_id,
            s1.final,
            s1.ranking,
            s1.calls,
            s1.replayed,
            failure=failed.failure,
        )
    asked = parse_ranking(reply.text)
    allowed = set(shown)
    order = tuple(e for e in dict.fromkeys(asked) if e in allowed)[:MAX_ANSWER]
    ranking = answer_scores(order, s1.ranking, world.catalog, hidden=_hidden(world, q))
    failure = s1.failure or ("no-json" if not asked else "")
    return Outcome(
        Scenario.S2,
        q.query.query_id,
        q.set_id,
        order,
        ranking,
        (*s1.calls, reply.record),
        s1.replayed + int(reply.replayed),
        failure=failure,
        detail={
            "review": reply.text,
            "outside_list": len(set(asked) - allowed),
            "shown": len(shown),
        },
    )


# --- S3 --------------------------------------------------------------------------------------


class _Loop:
    """S3's running state for one query."""

    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages
        self.calls: list[CallRecord] = []
        self.steps: list[Step] = []
        self.replayed = 0
        self.used = 0
        self.final: tuple[str, ...] | None = None
        self.final_as_text = False
        self.told_last = False
        self.nudges = 0


def run_s3(caller: Caller, world: World, q: QueryData) -> Outcome:
    session = Session(world, q)
    loop = _Loop(
        [
            _system(CONTEXT),
            _user(text_part(S3_TASK), image_part(q.query.image_id, LONG_SIDE)),
        ]
    )
    failure = ""
    try:
        for turn in range(MAX_TURNS):
            if _turn(caller, q, session, loop, turn):
                break
    except OpenRouterError as err:
        failure = f"api-error: {str(err)[:200]}"
        logger.warning("S3 %s: %s", q.query.query_id, err)
    if loop.final is None and not failure:
        failure = "no-final-answer"
    final = loop.final or ()
    ranking = answer_scores(
        (*final, *session.last_list), world.g5(q), world.catalog, hidden=_hidden(world, q)
    )
    tools_used = [s.tool for s in loop.steps if s.run]
    return Outcome(
        Scenario.S3,
        q.query.query_id,
        q.set_id,
        final,
        ranking,
        tuple(loop.calls),
        loop.replayed,
        failure=failure,
        detail={
            "tool_calls": len(tools_used),
            "final_as_text": loop.final_as_text,
            "nudges": loop.nudges,
            "last_list": list(session.last_list[:10]),
        },
        steps=tuple(loop.steps),
    )


def _turn(caller: Caller, q: QueryData, session: Session, loop: _Loop, turn: int) -> bool:
    """One model call and its tool calls. True when the search is over."""
    out_of_calls = loop.used >= MAX_TOOL_CALLS
    if out_of_calls and not loop.told_last:
        loop.messages.append(_user(text_part(S3_LAST_CALL)))
        loop.told_last = True
    reply = caller.ask(
        loop.messages,
        query_id=q.query.query_id,
        scenario=Scenario.S3,
        step=f"turn-{turn}",
        tools=TOOLS,
        tool_choice="required" if out_of_calls else "auto",
    )
    loop.calls.append(reply.record)
    loop.replayed += int(reply.replayed)
    if not reply.tool_calls:
        written = parse_ranking(reply.text, "entity_ids")
        if written:
            loop.final = session.final_answer(written).final
            loop.final_as_text = True
            return True
        loop.nudges += 1
        loop.messages.append({"role": "assistant", "content": reply.text})
        loop.messages.append(_user(text_part(S3_NUDGE)))
        return out_of_calls and loop.told_last and loop.nudges > 2
    loop.messages.append(_assistant(reply.text, reply.tool_calls, turn))
    thumbs: list[tuple[str, str, str]] = []
    for n, call in enumerate(reply.tool_calls):
        call_id = _call_id(call, turn, n)
        if call.name == Tool.FINAL_ANSWER:
            result = session.run(call.name, call.arguments)
            loop.final = result.final
            loop.messages.append(_tool(call_id, call.name, result.text))
            continue
        if loop.used >= MAX_TOOL_CALLS:
            refused = "not run: no tool calls left; call final_answer"
            loop.messages.append(_tool(call_id, call.name, refused))
            loop.steps.append(Step(turn, call.name, call.arguments, (), "", error=False, run=False))
            continue
        loop.used += 1
        result = session.run(call.name, call.arguments)
        mode = ""
        if result.filters is not None:
            mode = "none" if result.filters.empty else result.filters.mode.value
        loop.steps.append(
            Step(turn, call.name, call.arguments, result.listed, mode, result.error, run=True)
        )
        loop.messages.append(_tool(call_id, call.name, result.text))
        thumbs.extend(result.thumbnails)
    if loop.final is not None:
        return True
    if thumbs:
        parts: list[JSON] = [text_part("Thumbnails asked for with view_results:")]
        for label, _, image_id in thumbs:
            parts.append(text_part(label))
            parts.append(image_part(image_id, THUMB_PX))
        loop.messages.append(_user(*parts))
    return False


def _call_id(call: ToolCall, turn: int, n: int) -> str:
    # Mistral wants nine letters or digits; the provider's own ids already are.
    return call.id or f"t{turn:02d}c{n:02d}xyz"[:9]


def _assistant(text: str, calls: Sequence[ToolCall], turn: int) -> Message:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": _call_id(c, turn, n),
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for n, c in enumerate(calls)
        ],
    }


def _tool(call_id: str, name: str, text: str) -> Message:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": text}
