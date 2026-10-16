"""The conversation page (step H5): an open vision model leads the search with the visitor.

The visitor gives a photo, words, or both, then says in their own words what they want to
find. The site's usual search of the photo starts at once, while they type; its first
results go to the model with the goal. The model (Mistral Large 4 through OpenRouter, as in
H4) asks short questions, calls the tools of `miroir.web.chat_tools`, looks at thumbnails,
tries again, then shows its picks as the site's result cards. The visitor can keep writing,
and the model searches again.

A turn runs as a task on the server and writes events (`Conversation.events`); the page
reads them as a stream, which ends after `STREAM_S` (Modal cuts a request at 150 s) and is
asked for again from where it stopped.

Rules: only the visitor's own photo and thumbnails of publishable records go to the model
(`miroir.web.publishable`); each conversation and each day have a spending cap
(`Settings.chat_conversation_usd`, `chat_day_usd`); each paid call is logged with its cost.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import SecretStr

from miroir.images.fetch import to_jpeg
from miroir.jobs.openrouter import JSON, ChatAnswer, OpenRouterClient, OpenRouterError, ToolCall
from miroir.web.chat_tools import (
    MAX_RESULTS,
    Signature,
    ToolName,
    ToolOutcome,
    Wait,
    run_tool,
    tools_for,
)
from miroir.web.engine import SearchRequest, SearchResponse
from miroir.web.nearby import Position
from miroir.web.publishable import Publishable
from miroir.web.toolsearch import THUMB_SIDE, Hit, ToolHits, ToolQuery

logger = logging.getLogger(__name__)

#: The model, as in H4: open, takes images and tools; reasoning off, temperature 0.
MODEL = "mistralai/mistral-large-4-0"
MODEL_FIELDS: dict[str, JSON] = {"temperature": 0.0, "reasoning": {"effort": "none"}}
MAX_TOKENS = 800
#: Searches per visitor message; then the model must show results or ask.
MAX_TOOL_CALLS = 8
#: Looks at thumbnails per visitor message, apart from the searches (the smoke test with
#: Mistral spent its 8 calls on searches and could not check its picks by eye).
MAX_VIEWS = 3
#: Model calls per visitor message, all told.
MAX_MODEL_CALLS = 12
#: Thumbnails kept in what the model is sent again each call; older ones become a note.
KEEP_THUMBS = 12
#: Long side of the visitor's photo and framed region as the model sees them.
PHOTO_SIDE = 768
#: First-look results the model is told about.
FIRST_LOOK = 10
#: The cost guessed for a first call, before one is known.
FIRST_CALL_USD = 0.01
#: A stream request ends after this many seconds; the page asks again.
STREAM_S = 110.0
#: The stream says it is alive this often while nothing happens.
PING_S = 10.0
#: The first look may wait this long for the search engine to start.
FIRST_LOOK_WAIT_S = 240.0
#: Conversations are forgotten after this long without a message, and at most this many kept.
IDLE_S = 2 * 3600.0
MAX_CONVERSATIONS = 200
#: Longest message the visitor may write.
MAX_WORDS = 500


class ChatModel(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        """One chat completion; `body` holds messages, tools and the other request fields."""
        ...


class OpenRouterModel:
    """Mistral Large 4 through OpenRouter. The key comes from `Settings`, never logged."""

    def __init__(self, api_key: SecretStr, model: str = MODEL) -> None:
        self._client = OpenRouterClient(api_key, timeout_s=90.0)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        return await asyncio.to_thread(self._client.chat, self._model, {**body, **MODEL_FIELDS})


class SpendStore(Protocol):
    """Dollars spent per UTC day, shared by every conversation."""

    async def spent(self, day: str) -> float: ...
    async def add(self, day: str, usd: float) -> float:
        """Adds and returns the day's new total."""
        ...


class MemorySpend:
    def __init__(self) -> None:
        self._days: dict[str, float] = {}

    async def spent(self, day: str) -> float:
        return self._days.get(day, 0.0)

    async def add(self, day: str, usd: float) -> float:
        self._days[day] = self._days.get(day, 0.0) + usd
        return self._days[day]


class FileSpend:
    """Per-day totals in a small JSON file, so a restarted `miroir web dev` keeps the day's."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _days(self) -> dict[str, float]:
        if not self._path.exists():
            return {}
        raw: object = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {str(k): float(str(v)) for k, v in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

    async def spent(self, day: str) -> float:
        return self._days().get(day, 0.0)

    async def add(self, day: str, usd: float) -> float:
        days = self._days()
        days[day] = days.get(day, 0.0) + usd
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(days, indent=1) + "\n", encoding="utf-8")
        return days[day]


@dataclass(frozen=True, slots=True)
class Limits:
    conversation_usd: float
    day_usd: float


class Backend(Protocol):
    """What the conversation needs of the search backend (`miroir.web.app.SearchBackend`)."""

    async def search(self, request: SearchRequest) -> SearchResponse: ...
    async def tool_search(self, query: ToolQuery) -> ToolHits: ...


class ThumbSource:
    """Thumbnails the model may see: each id checked as publishable before any byte is read."""

    def __init__(self, publishable: Publishable, fetch: Callable[[str], Awaitable[bytes]]) -> None:
        self.publishable = publishable
        self._fetch = fetch
        self._cache: dict[str, bytes] = {}

    async def jpeg(self, image_id: str, url: str) -> bytes | None:
        self.publishable.check(image_id, "the model")
        if image_id in self._cache:
            return self._cache[image_id]
        try:
            data = await self._fetch(url)
            small = await asyncio.to_thread(to_jpeg, data, THUMB_SIDE)
        except (httpx.HTTPError, OSError, ValueError) as err:
            logger.info("thumbnail %s not fetched: %s", image_id, type(err).__name__)
            return None
        if len(self._cache) > 2000:
            self._cache.clear()
        self._cache[image_id] = small
        return small


class Visit(StrEnum):
    """What the visitor sends."""

    GOAL = "goal"
    """The first message: what they want to find."""
    ANSWER = "answer"
    REGION = "region"
    MESSAGE = "message"


@dataclass(frozen=True, slots=True)
class VisitorInput:
    kind: Visit
    text: str = ""
    region: bytes | None = None
    """The framed part of the photo; None when they framed nothing."""
    position: Position | None = None


@dataclass(frozen=True, slots=True)
class CardView:
    """A result card, as the page draws it (the site's card, plus the model's reason)."""

    entity_id: str
    title: str
    place: str
    collection: str
    collection_label: str
    pop_url: str
    image_url: str | None
    image_fallback_url: str | None
    matched: str | None
    photo_count: int
    level: str | None = None
    level_label: str | None = None
    reason: str = ""


class BusyError(RuntimeError):
    """A message arrived while the model is still working on the previous one."""


Event = dict[str, JSON]


@dataclass(frozen=True, slots=True)
class _Pending:
    kind: Wait
    message: int
    """Index of the tool message waiting for the visitor's reply."""


class Conversation:
    """One visitor's conversation: what the model has been told and seen, and what it cost."""

    def __init__(
        self,
        service: ChatService,
        *,
        session: str,
        photo: bytes | None,
        words: str,
        position: Position | None,
        now: float,
    ) -> None:
        self.service = service
        self.id = secrets.token_urlsafe(16)
        self.session = session
        self.photo_bytes = photo
        self.words = words
        self.position_given = position
        self.touched = now
        self.messages: list[dict[str, JSON]] = []
        self.events: list[Event] = []
        self.changed = asyncio.Event()
        self.first_look: asyncio.Future[SearchResponse] | None = None
        self.task: asyncio.Future[None] | None = None
        self.pending: _Pending | None = None
        self.started = False
        self.known: dict[str, CardView] = {}
        """Every entity the model has seen in a result: what show_results may show."""
        self.signatures: list[tuple[Signature, int]] = []
        self.region_bytes: bytes | None = None
        self.thumbs: dict[str, bytes] = {}
        """Thumbnails fetched for the model, by image id (each one publishable)."""
        self.spent_usd = 0.0
        self.last_call_usd = FIRST_CALL_USD
        self.calls = 0
        self.steps = 0
        self.views = 0
        """Looks at thumbnails in the current turn."""

    # --- ToolContext ---------------------------------------------------------------------------

    @property
    def photo(self) -> bytes | None:
        return self.photo_bytes

    @property
    def region(self) -> bytes | None:
        return self.region_bytes

    @property
    def position(self) -> Position | None:
        return self.position_given

    async def search(self, query: ToolQuery) -> ToolHits:
        return await self.service.backend.tool_search(query)

    def remember(self, hits: Sequence[Hit]) -> None:
        for hit in hits:
            self.known.setdefault(hit.entity_id, _card_of_hit(hit))

    def seen(self, entity_id: str) -> bool:
        return entity_id in self.known

    async def thumbnail(self, hit: Hit) -> str | None:
        source = self.service.thumbs
        for image in hit.images:
            if not source.publishable.allows(image.image_id):
                continue
            data = await source.jpeg(image.image_id, image.url)
            if data is not None:
                self.thumbs[image.image_id] = data
                return image.image_id
        return None

    def repeats(self, signature: Signature) -> int | None:
        for earlier, step in self.signatures:
            if signature.same(earlier):
                return step
        self.signatures.append((signature, self.steps))
        return None

    # --- events --------------------------------------------------------------------------------

    def emit(self, event: Event) -> None:
        self.events.append(event)
        done, self.changed = self.changed, asyncio.Event()
        done.set()

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


class ChatService:
    """Every open conversation, and the means they share."""

    def __init__(
        self,
        *,
        backend: Backend,
        model: ChatModel,
        thumbs: ThumbSource,
        spend: SpendStore,
        limits: Limits,
        index_size: tuple[int, int],
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], str] | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.thumbs = thumbs
        self.spend = spend
        self.limits = limits
        self.index_size = index_size
        """(images, entities) the index holds, for the brief."""
        self._clock = clock
        self._today = today or (lambda: datetime.now(UTC).date().isoformat())
        self._conversations: dict[str, Conversation] = {}

    # --- conversations -------------------------------------------------------------------------

    def open(
        self, session: str, *, photo: bytes | None, words: str, position: Position | None
    ) -> Conversation:
        """A new conversation. With a photo, the site's usual search starts at once."""
        self._forget_old()
        conversation = Conversation(
            self,
            session=session,
            photo=photo,
            words=words.strip()[:MAX_WORDS],
            position=position,
            now=self._clock(),
        )
        if photo is not None:
            request = SearchRequest(image=photo, text=conversation.words, position=position)
            conversation.first_look = asyncio.ensure_future(self.backend.search(request))
        self._conversations[conversation.id] = conversation
        logger.info(
            "chat %s opened: photo=%s words=%d", conversation.id[:8], photo is not None, len(words)
        )
        return conversation

    def get(self, session: str, conversation_id: str) -> Conversation | None:
        found = self._conversations.get(conversation_id)
        if found is None or found.session != session:
            return None
        return found

    def send(self, conversation: Conversation, given: VisitorInput) -> int:
        """Start the model's turn on what the visitor sent. Returns the first new event's
        number, to read the stream from."""
        if conversation.running:
            msg = "the model is still working"
            raise BusyError(msg)
        conversation.touched = self._clock()
        after = len(conversation.events)
        conversation.task = asyncio.ensure_future(self._turn(conversation, given))
        return after

    async def stream(self, conversation: Conversation, after: int) -> AsyncIterator[Event]:
        """Events from number `after` on, until the turn ends or `STREAM_S` pass."""
        deadline = self._clock() + STREAM_S
        n = max(after, 0)
        while True:
            waiter = conversation.changed
            while n < len(conversation.events):
                yield {**conversation.events[n], "n": n}
                n += 1
            if not conversation.running:
                return
            left = deadline - self._clock()
            if left <= 0:
                yield {"type": "continue", "after": n}
                return
            try:
                await asyncio.wait_for(waiter.wait(), min(PING_S, left))
            except TimeoutError:
                yield {"type": "ping"}

    def _forget_old(self) -> None:
        now = self._clock()
        for key, c in list(self._conversations.items()):
            if now - c.touched > IDLE_S and not c.running:
                del self._conversations[key]
        while len(self._conversations) >= MAX_CONVERSATIONS:
            idle = [c for c in self._conversations.values() if not c.running]
            if not idle:
                break
            del self._conversations[min(idle, key=lambda c: c.touched).id]

    # --- one turn ------------------------------------------------------------------------------

    async def _turn(self, c: Conversation, given: VisitorInput) -> None:
        waiting = "message"
        try:
            await self._take(c, given)
            waiting = await self._loop(c)
        except Exception:
            logger.exception("chat %s: turn failed", c.id[:8])
            c.emit({"type": "error", "text": "La conversation a rencontré une erreur."})
        c.emit({"type": "end", "waiting": waiting})

    async def _take(self, c: Conversation, given: VisitorInput) -> None:
        """Add what the visitor sent to the messages."""
        if given.position is not None:
            c.position_given = given.position
        text = given.text.strip()[:MAX_WORDS]
        if not c.started:
            c.started = True
            c.messages = [{"role": "system", "content": self._brief()}]
            c.messages.append(await self._opening(c, text))
            return
        shared = ""
        if given.position is not None:
            shared = (
                f" They shared their position (lat {given.position.latitude:.5f}, lon "
                f'{given.position.longitude:.5f}): filters near "visitor" uses it.'
            )
        pending, c.pending = c.pending, None
        if pending is None:
            c.messages.append(_user(_text(f"The visitor writes: «{text}»{shared}")))
            return
        if pending.kind is Wait.REGION and given.kind is Visit.REGION:
            if given.region is None:
                note = "The visitor framed nothing: the whole photo is what they mean."
                c.messages[pending.message]["content"] = note
                return
            c.region_bytes = given.region
            c.messages[pending.message]["content"] = (
                "The visitor framed a part of the photo; it follows. image_search with source "
                "region searches it."
            )
            c.messages.append(
                _user(_text("The part the visitor framed:"), _image_ref("region", len(c.thumbs)))
            )
            return
        answer = text or "je ne sais pas"
        c.messages[pending.message]["content"] = f"The visitor answers: «{answer}».{shared}"

    async def _opening(self, c: Conversation, goal: str) -> dict[str, JSON]:
        parts: list[JSON] = [_text(f"The visitor's goal, in their own words: «{goal or '?'}»")]
        if c.photo_bytes is None:
            parts.append(
                _text(f"No photo: the visitor only typed words: «{c.words}». Search with them.")
            )
            return _user(*parts)
        parts += [_text("The visitor's photo:"), _image_ref("photo", 0)]
        if c.words:
            parts.append(_text(f"Words the visitor typed with the photo: «{c.words}»"))
        if c.position_given is not None:
            parts.append(
                _text('The photo was taken at a known position: filters near "visitor" uses it.')
            )
        parts.append(_text(await self._first_look(c)))
        return _user(*parts)

    async def _first_look(self, c: Conversation) -> str:
        """The site's usual search, started when the conversation opened."""
        assert c.first_look is not None
        c.steps += 1
        step = c.steps
        label = "Première recherche par image (la recherche habituelle)"
        c.emit({"type": "step", "id": step, "state": "running", "label": label})
        try:
            found = await asyncio.wait_for(asyncio.shield(c.first_look), FIRST_LOOK_WAIT_S)
        except (TimeoutError, ValueError, RuntimeError) as err:
            logger.warning("chat %s: first look failed: %s", c.id[:8], err)
            c.emit(_step_event(step, label, "pas de résultat", (), error=True))
            return "First look: the usual search did not answer. Search with the tools."
        cards = found.cards[:FIRST_LOOK]
        for card in found.cards:
            c.known.setdefault(
                card.entity_id,
                CardView(
                    card.entity_id,
                    card.title,
                    card.place,
                    card.collection,
                    card.collection_label,
                    card.pop_url,
                    card.image_url,
                    card.image_fallback_url,
                    card.matched,
                    card.photo_count,
                    card.level.value,
                    card.level_label,
                ),
            )
        thumbs: list[JSON] = [
            {"entity_id": k.entity_id, "title": k.title, "image_url": k.image_url}
            for k in cards[:8]
        ]
        c.emit(_step_event(step, label, f"{len(found.cards)} résultats", thumbs))
        lines = [
            f"{i}. {k.entity_id} | {k.collection} | {k.title} | {k.place} | {k.level_label}"
            for i, k in enumerate(cards, start=1)
        ]
        head = "First look (the site's usual search for this photo; it may be wrong):"
        if found.caption:
            head += f"\nCaption the site read on the print: «{found.caption}»"
        return head + "\n" + ("\n".join(lines) if lines else "nothing found")

    async def _loop(self, c: Conversation) -> str:
        """Model calls and their tools until the model shows results, asks, or stops."""
        tools = tools_for(photo=c.photo_bytes is not None)
        c.views = 0
        schemas = [t.schema() for t in tools]
        offered = {t.name.value for t in tools}
        used = 0
        told_last = False
        nudged = False
        for _ in range(MAX_MODEL_CALLS):
            stop = await self._over_limit(c)
            if stop is not None:
                c.emit({"type": "limit", "text": stop})
                return "limit"
            out_of_calls = used >= MAX_TOOL_CALLS
            if out_of_calls and not told_last:
                c.messages.append(_user(_text(LAST_CALL)))
                told_last = True
            body: dict[str, JSON] = {
                "messages": self._render(c),
                "tools": schemas,
                "tool_choice": "required" if out_of_calls else "auto",
                "max_tokens": MAX_TOKENS,
            }
            c.emit({"type": "status", "text": "Le modèle réfléchit…"})
            try:
                answer = await self.model.complete(body)
            except OpenRouterError as err:
                logger.warning("chat %s: model call failed: %s", c.id[:8], err)
                c.emit({"type": "error", "text": MODEL_DOWN})
                return "message"
            await self._paid(c, answer.cost_usd)
            said = answer.text.strip()
            if said:
                c.emit({"type": "say", "text": said})
            if not answer.tool_calls:
                c.messages.append({"role": "assistant", "content": answer.text})
                if nudged:
                    return "message"
                # Mistral sometimes says what it will do next and stops: once a turn, it is
                # asked to do it.
                c.messages.append(_user(_text(NUDGE)))
                nudged = True
                continue
            ids = [_call_id(call, c.calls, n) for n, call in enumerate(answer.tool_calls)]
            c.messages.append(_assistant(answer.text, answer.tool_calls, ids))
            waiting, thumbs, used = await self._calls(c, answer.tool_calls, ids, offered, used)
            if thumbs:
                parts: list[JSON] = [_text("Thumbnails asked for with view_results:")]
                for label, image_id in thumbs:
                    parts += [_text(label), _image_ref("thumb", image_id)]
                c.messages.append(_user(*parts))
            if waiting is not None:
                return waiting
        c.emit({"type": "say", "text": GAVE_UP})
        return "message"

    async def _calls(
        self,
        c: Conversation,
        calls: Sequence[ToolCall],
        ids: Sequence[str],
        offered: set[str],
        used: int,
    ) -> tuple[str | None, list[tuple[str, str]], int]:
        """Run one reply's tool calls. `used` counts the turn's searches; views are counted
        in `Conversation.views`."""
        views = c.views
        waiting: str | None = None
        thumbs: list[tuple[str, str]] = []
        for call, call_id in zip(calls, ids, strict=True):
            if waiting is not None:
                c.messages.append(_tool(call_id, call.name, NOT_RUN_WAITING))
                continue
            if call.name not in offered:
                text = f"error: {call.name} is not available here (no photo)."
                c.messages.append(_tool(call_id, call.name, text))
                continue
            counted = call.name in COUNTED
            if counted and used >= MAX_TOOL_CALLS:
                c.messages.append(_tool(call_id, call.name, NOT_RUN_CAP))
                logger.info("chat %s: %s refused, no searches left", c.id[:8], call.name)
                continue
            if call.name == ToolName.VIEW_RESULTS.value:
                if views >= MAX_VIEWS:
                    c.messages.append(_tool(call_id, call.name, NOT_RUN_VIEWS))
                    continue
                views += 1
            c.steps += 1
            step = c.steps
            if counted:
                c.emit(
                    {"type": "step", "id": step, "state": "running", "label": RUNNING[call.name]}
                )
            outcome = await run_tool(c, call.name, call.arguments)
            if outcome.counted and counted:
                used += 1
            logger.info(
                "chat %s step %d: %s%s",
                c.id[:8],
                step,
                call.name,
                " (error)" if outcome.error else "",
            )
            waiting = self._outcome(c, outcome, step, call_id, call.name)
            thumbs += [(t.label, t.image_id) for t in outcome.thumbnails]
        c.views = views
        return waiting, thumbs, used

    def _outcome(
        self, c: Conversation, outcome: ToolOutcome, step: int, call_id: str, name: str
    ) -> str | None:
        """Put one tool's outcome in the messages and the events; what to wait for, if any."""
        if outcome.wait is not None:
            c.messages.append(_tool(call_id, name, "(waiting for the visitor)"))
            c.pending = _Pending(outcome.wait, len(c.messages) - 1)
            if outcome.question is not None:
                q = outcome.question
                c.emit(
                    {
                        "type": "question",
                        "text": q.text,
                        "choices": list(q.choices),
                        "position": q.offer_position,
                    }
                )
                return "answer"
            c.emit({"type": "region", "text": outcome.region_reason})
            return "region"
        c.messages.append(_tool(call_id, name, outcome.text))
        if outcome.picks is not None:
            cards: list[JSON] = []
            for pick in outcome.picks[:MAX_RESULTS]:
                known = c.known[pick.entity_id]
                cards.append(_card_json(known, pick.reason))
            c.emit({"type": "results", "cards": cards})
            return "message"
        if outcome.step is not None:
            s = outcome.step
            thumbs: list[JSON] = [
                {"entity_id": t.entity_id, "title": t.title, "image_url": t.image_url}
                for t in s.thumbs
            ]
            c.emit(_step_event(step, s.label, s.summary, thumbs, error=s.error))
        return None

    # --- money ---------------------------------------------------------------------------------

    async def _over_limit(self, c: Conversation) -> str | None:
        """Why the next call may not be made, or None. It is refused when it could take the
        conversation or the day past its cap, guessing its cost from the last call's."""
        guess = c.last_call_usd
        if c.spent_usd + guess > self.limits.conversation_usd:
            logger.info("chat %s: conversation cap reached at $%.4f", c.id[:8], c.spent_usd)
            return CONVERSATION_CAP.format(cap=_dollars(self.limits.conversation_usd))
        if await self.spend.spent(self._today()) + guess > self.limits.day_usd:
            logger.info("chat %s: day cap reached", c.id[:8])
            return DAY_CAP
        return None

    async def _paid(self, c: Conversation, usd: float) -> None:
        c.calls += 1
        c.spent_usd += usd
        if usd > 0:  # the next call sends a little more: guess it costs what this one did
            c.last_call_usd = usd
        day = await self.spend.add(self._today(), usd)
        logger.info(
            "chat %s call %d: $%.4f (conversation $%.4f, today $%.4f)",
            c.id[:8],
            c.calls,
            usd,
            c.spent_usd,
            day,
        )

    # --- what the model is sent ----------------------------------------------------------------

    def _brief(self) -> str:
        images, entities = self.index_size
        return BRIEF.format(
            images=f"{images:,}",
            entities=f"{entities:,}",
            calls=MAX_TOOL_CALLS,
            views=MAX_VIEWS,
        )

    def _render(self, c: Conversation) -> list[JSON]:
        """The messages with images as data URLs. Only the last `KEEP_THUMBS` thumbnails go
        again; each is checked as publishable once more before it leaves."""
        thumb_parts = [
            (i, j)
            for i, m in enumerate(c.messages)
            for j, p in enumerate(_parts(m))
            if isinstance(p, dict) and p.get("type") == "image_ref" and p.get("ref") == "thumb"
        ]
        keep = set(thumb_parts[-KEEP_THUMBS:])
        out: list[JSON] = []
        for i, message in enumerate(c.messages):
            parts = _parts(message)
            if not parts:
                out.append(dict(message))
                continue
            rendered: list[JSON] = []
            for j, part in enumerate(parts):
                if not (isinstance(part, dict) and part.get("type") == "image_ref"):
                    rendered.append(part)
                    continue
                ref = str(part.get("ref"))
                if ref == "thumb":
                    image_id = str(part.get("image_id"))
                    if (i, j) not in keep or image_id not in c.thumbs:
                        rendered.append(_text("(thumbnail shown earlier)"))
                        continue
                    self.thumbs.publishable.check(image_id, "the model")
                    rendered.append(_data_url(c.thumbs[image_id]))
                elif ref == "region" and c.region_bytes is not None:
                    rendered.append(_data_url(to_jpeg(c.region_bytes, PHOTO_SIDE)))
                elif ref == "photo" and c.photo_bytes is not None:
                    rendered.append(_data_url(to_jpeg(c.photo_bytes, PHOTO_SIDE)))
            out.append({**message, "content": rendered})
        return out


COUNTED = {ToolName.IMAGE_SEARCH.value, ToolName.TEXT_SEARCH.value}
RUNNING = {
    ToolName.IMAGE_SEARCH.value: "Recherche par image…",
    ToolName.TEXT_SEARCH.value: "Recherche par mots…",
    ToolName.VIEW_RESULTS.value: "Regarde des images de près…",
}

LAST_CALL = (
    f"You have used your {MAX_TOOL_CALLS} searches for this message. Call show_results now "
    "with the best you have, or ask_visitor one question."
)
NOT_RUN_CAP = "not run: no searches left for this message. Call show_results, or ask_visitor."
NOT_RUN_VIEWS = (
    f"not run: {MAX_VIEWS} looks at thumbnails already for this message. Call show_results."
)
NUDGE = (
    "(Not from the visitor.) You wrote without calling a tool. If you meant to search, call "
    "the tool now; to ask the visitor, call ask_visitor; to finish, call show_results. If you "
    "only meant to answer the visitor, write nothing more."
)
NOT_RUN_WAITING = "not run: you are waiting for the visitor. Call it again after their answer."
MODEL_DOWN = (
    "Le modèle ne répond pas pour le moment. Réessayez dans un instant, ou utilisez la "
    "recherche habituelle."
)
GAVE_UP = (
    "Je m’arrête là pour ce message : je n’ai pas trouvé mieux. Dites-m’en plus, ou utilisez la "
    "recherche habituelle."
)
CONVERSATION_CAP = (
    "Cette conversation a atteint sa limite de dépense ({cap}). Lancez la recherche habituelle "
    "avec votre photo, ou commencez une nouvelle conversation."
)
DAY_CAP = (
    "La recherche en conversation a atteint sa limite de dépense pour aujourd’hui. La recherche "
    "habituelle reste ouverte."
)

BRIEF = """You help a visitor find French heritage records (the POP database) from a photo, \
words, or both. You lead the search: ask a short question when it helps, call the search \
tools, look at what comes back, try again, then show your picks.

The index holds {images} images of {entities} buildings and objects. Each has an id, a base \
and images:
- merimee: buildings (ids PA...). Archive prints from the Médiathèque du patrimoine, mostly \
black and white, often glued on a card with a typed caption naming the commune and the \
building; and recent colour photos from Wikimedia Commons.
- palissy: objects kept in churches and other buildings (ids PM...), photographed in the \
same archive.
- joconde: museum objects, often several views of one object.
- rose-valland: works looted during the war (MNR), and Jeu de Paume room photos.
- memoire: archive photos tied to no building or object.

The visitor says first what they want to find, in their words. Let it set how you search:
- One exact building or object: find that one record. Show 1 to 5 results, only ones that \
could be it, the best first.
- Similar things: the look counts more than the place. Show up to 10 that share the kind, \
style or subject.
- The same place: find where the photo was taken, then what the index holds there \
(commune filter, near).
- Who made it, where it comes from: find the record, then say in one or two sentences what \
the records give (author, date, place), and show it.
- Anything else: do what they ask.

Questions. Ask only what would narrow the search for this photo, and only when the photo, \
its words and the first look leave you unsure. When the first look and one more search have \
not settled it, ask before you guess: one answer from the visitor beats many searches, and \
never search a name you only guessed. One question at a time, at most 3 in a conversation. \
Fit the question to the subject: a building, a painting, an object, an archive \
print each call for different things (the town or the street; the museum or the room; what \
the object is or where it is kept; the caption you cannot read). Give 2 to 4 short suggested \
answers; the visitor can also write their own or say they do not know. Set offer_position \
only when where the visitor stands now would help. When the photo shows several things (a \
wall of paintings, a room), ask_region lets the visitor frame the one they mean.

Searching:
- The first look is the site's usual search for the photo. It may already be right.
- image_search model auto is the site's tuned choice for this photo. A7 suits black and white \
archive prints of buildings; A3 suits colour photos of buildings and objects.
- Read any text in the photo (a typed caption, a label, a sign) and search its words with \
text_search. Write French words as the records do (église, portail, retable, the commune's \
name). This works very well on archive prints.
- Filters favour matching records by default (boost). Use mode hard only when you are sure (the \
visitor said so, or a caption names the commune): a hard filter on a wrong guess hides the \
right answer.
- Do not run a search again with reworded words: it will be refused. If two searches fail, \
change approach: another image model, a filter, view_results to compare by eye, a question, \
or show what you have.
- At most {calls} searches and {views} looks at thumbnails per visitor message.
- Before you show picks for an exact match, check them by eye with view_results. Do not say \
a pick looks like the photo unless you have seen its thumbnail.

End each turn with show_results: ids you have seen in results, best first, each with one short \
reason in French ("même portail roman, même commune"). If nothing fits, say so in one short \
sentence and show the closest anyway, or ask one question. The visitor may write again \
("pas celle-ci, l'église est plus ancienne"): search again with what they say.

Everything you write outside a tool call is shown to the visitor: write it in French, in one \
or two short plain sentences, without database words (no "base", "entité", "score", "id"), \
and never in English. Never invent a name, a date or a place the records do not give."""


# --- message helpers ---------------------------------------------------------------------------


def _text(text: str) -> JSON:
    return {"type": "text", "text": text}


def _image_ref(ref: str, image_id: object) -> JSON:
    """An image named, not held: it becomes a data URL only when sent."""
    return {"type": "image_ref", "ref": ref, "image_id": str(image_id)}


def _data_url(jpeg: bytes) -> JSON:
    url = f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


def _user(*parts: JSON) -> dict[str, JSON]:
    return {"role": "user", "content": list(parts)}


def _parts(message: Mapping[str, JSON]) -> list[JSON]:
    content = message.get("content")
    return content if isinstance(content, list) else []


def _assistant(text: str, calls: Sequence[ToolCall], ids: Sequence[str]) -> dict[str, JSON]:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call, call_id in zip(calls, ids, strict=True)
        ],
    }


def _tool(call_id: str, name: str, text: str) -> dict[str, JSON]:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": text}


def _call_id(call: ToolCall, turn: int, n: int) -> str:
    # Mistral wants nine letters or digits; the provider's own ids already are.
    return call.id or f"c{turn:03d}n{n:02d}xy"[:9]


def _step_event(
    step: int, label: str, summary: str, thumbs: Sequence[JSON], *, error: bool = False
) -> Event:
    return {
        "type": "step",
        "id": step,
        "state": "done",
        "label": label,
        "summary": summary,
        "thumbs": list(thumbs),
        "error": error,
    }


def _card_of_hit(hit: Hit) -> CardView:
    return CardView(
        hit.entity_id,
        hit.title,
        hit.place,
        hit.collection,
        hit.collection_label,
        hit.pop_url,
        hit.image_url,
        hit.image_fallback_url,
        hit.matched,
        hit.photo_count,
    )


def _card_json(card: CardView, reason: str) -> JSON:
    return {
        "entity_id": card.entity_id,
        "title": card.title,
        "place": card.place,
        "collection": card.collection,
        "collection_label": card.collection_label,
        "pop_url": card.pop_url,
        "image_url": card.image_url,
        "image_fallback_url": card.image_fallback_url,
        "matched": card.matched,
        "photo_count": card.photo_count,
        "level": card.level,
        "level_label": card.level_label,
        "reason": reason,
    }


def _dollars(usd: float) -> str:
    return f"{usd:.2f} $".replace(".", ",")
