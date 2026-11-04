"""The conversation page (step H5): an open vision model leads the search with the visitor.

The visitor gives a photo, words, or both, then says in their own words what they want to
find. The site's usual search of the photo starts at once, while they type; its first
results go to the model with the goal. The model (`miroir.web.chat_model`) asks short
questions, calls the tools of `miroir.web.chat_tools`, reads records, looks at thumbnails,
tries again, then shows its picks as the site's result cards. The visitor can keep writing,
and the model searches again.

A turn runs as a task on the server and writes events (`Conversation.events`); the page
reads them as a stream, which ends after `STREAM_S` (so no request stays open for long) and is
asked for again from where it stopped.

Rules: only the visitor's own photo and thumbnails of publishable records go to the model
(`miroir.web.publishable`); each conversation and each day have a spending cap
(`miroir.web.chat_spend`); each paid call is logged with its cost. The search method is
enforced by `miroir.web.chat_method`, the messages are built by `miroir.web.chat_render`,
and every word said is in `miroir.web.chat_prompts`.

Each conversation is written down as JSON lines (`miroir.web.chatlog`), without images.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

import httpx

from miroir.images.fetch import to_jpeg
from miroir.jobs.mistral import MistralError
from miroir.jobs.openrouter import JSON, OpenRouterError, ToolCall
from miroir.outlines import DETECT_SIDE, Box, Outlines
from miroir.web import chat_method, chat_prompts
from miroir.web.catalog import Catalog, CatalogRecord, Filters
from miroir.web.chat_method import COUNTED, Method
from miroir.web.chat_model import ChatModel, MistralModel, OpenRouterModel, chat_model
from miroir.web.chat_prompts import BRIEF, LOCKED_NOTE, ONESHOT_NOTE
from miroir.web.chat_render import (
    CardView,
    Event,
    Message,
    assistant_message,
    call_id_of,
    card_json,
    card_of_hit,
    image_ref,
    render,
    shown_images,
    step_event,
    summarised,
    text_part,
    tool_message,
    user_message,
    with_note,
)
from miroir.web.chat_spend import FileSpend, Limits, MemorySpend, SpendStore
from miroir.web.chat_tools import (
    MAX_RESULTS,
    OUTLINES_LISTED,
    Signature,
    ToolName,
    ToolOutcome,
    Wait,
    arguments_of,
    run_tool,
    running_label,
    tools_for,
    why_of,
)
from miroir.web.chatlog import ChatLog, NoLog, redact
from miroir.web.engine import SearchRequest, SearchResponse
from miroir.web.levels import Level
from miroir.web.nearby import Position
from miroir.web.publishable import Publishable
from miroir.web.toolsearch import THUMB_SIDE, Hit, ToolHits, ToolQuery

__all__ = [
    "BRIEF",
    "LOCKED_NOTE",
    "ONESHOT_NOTE",
    "BusyError",
    "CardView",
    "ChatModel",
    "ChatService",
    "Conversation",
    "FileSpend",
    "Limits",
    "MemorySpend",
    "MistralModel",
    "OpenRouterModel",
    "SpendStore",
    "ThumbSource",
    "Visit",
    "VisitorInput",
    "chat_model",
]

logger = logging.getLogger(__name__)

MAX_TOKENS = 800
#: Model calls per visitor message, all told.
MAX_MODEL_CALLS = 25
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


class Backend(Protocol):
    """What the conversation needs of the search backend (`miroir.web.app.SearchBackend`)."""

    async def search(self, request: SearchRequest) -> SearchResponse: ...
    async def tool_search(self, query: ToolQuery) -> ToolHits: ...
    async def detect(self, jpeg: bytes) -> Outlines:
        """The works in a photo (F4), as the photo view outlines them."""
        ...


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


class BusyError(RuntimeError):
    """A message arrived while the model is still working on the previous one."""


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
        oneshot: bool = False,
        filters: Filters | None = None,
    ) -> None:
        self.service = service
        self.id = secrets.token_urlsafe(16)
        self.session = session
        self.photo_bytes = photo
        self.words = words
        self.position_given = position
        self.oneshot = oneshot
        """The page's "Approfondi" search: one turn, no question to the visitor."""
        self.filters = filters or Filters()
        """The filters the visitor set on the page."""
        self.touched = now
        self.messages: list[Message] = []
        self.events: list[Event] = []
        self.changed = asyncio.Event()
        self.first_look: asyncio.Future[SearchResponse] | None = None
        self.detected: asyncio.Future[Outlines] | None = None
        """The detector's outlines of the photo, started with the first look."""
        self.query_images: dict[str, bytes] = {}
        """Transformed photos the model searched, kept for it to look at, by key."""
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
        self.day = service.today()
        """The day the conversation opened: its log file's folder."""
        self.viewed: dict[str, str] = {}
        """Entities the model has looked at, and the image it saw of each."""
        self.method = Method()

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
            self.known.setdefault(hit.entity_id, card_of_hit(hit))

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
                self.viewed[hit.entity_id] = image.image_id
                return image.image_id
        return None

    def viewed_image(self, entity_id: str) -> str | None:
        return self.viewed.get(entity_id)

    def checked(self, entity_id: str) -> bool:
        return entity_id in self.viewed or entity_id in self.method.sure

    async def outlines(self) -> Sequence[Box]:
        if self.detected is None:
            return ()
        try:
            found = await asyncio.wait_for(asyncio.shield(self.detected), FIRST_LOOK_WAIT_S)
        except (TimeoutError, ValueError, RuntimeError, OSError) as err:
            logger.warning("chat %s: outlines failed: %s", self.id[:8], err)
            return ()
        return found.boxes[:OUTLINES_LISTED]

    def keep_query_image(self, jpeg: bytes) -> str:
        key = f"query-{len(self.query_images) + 1}"
        self.query_images[key] = jpeg
        return key

    def title_of(self, entity_id: str) -> str:
        known = self.known.get(entity_id)
        return "" if known is None else known.title

    def records(self, entity_id: str) -> Sequence[CatalogRecord]:
        catalog = self.service.catalog
        if catalog is None:
            return ()
        row = catalog.entity_row.get(entity_id)
        return () if row is None else catalog.entity_records(row)

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

    def log(self, line: Mapping[str, JSON]) -> None:
        self.service.log.write(self.day, self.id, line)

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
        catalog: Catalog | None = None,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], str] | None = None,
        log: ChatLog | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.thumbs = thumbs
        self.spend = spend
        self.limits = limits
        self.index_size = index_size
        """(images, entities) the index holds, for the brief."""
        self.catalog = catalog
        """Record text for read_record; None reads nothing."""
        self._clock = clock
        self._today = today or (lambda: datetime.now(UTC).date().isoformat())
        self._conversations: dict[str, Conversation] = {}
        self.log: ChatLog = log or NoLog()

    def today(self) -> str:
        return self._today()

    # --- conversations -------------------------------------------------------------------------

    def open(
        self,
        session: str,
        *,
        photo: bytes | None,
        words: str,
        position: Position | None,
        oneshot: bool = False,
        filters: Filters | None = None,
    ) -> Conversation:
        """A new conversation. With a photo, the site's usual search starts at once.
        `oneshot`: the page's "Approfondi" search, which asks the visitor nothing; `filters`:
        those the visitor set, which the first look applies and the model is told about."""
        self._forget_old()
        conversation = Conversation(
            self,
            session=session,
            photo=photo,
            words=words.strip()[:MAX_WORDS],
            position=position,
            now=self._clock(),
            oneshot=oneshot,
            filters=filters,
        )
        if photo is not None:
            request = SearchRequest(
                image=photo,
                text=conversation.words,
                position=position,
                filters=conversation.filters,
            )
            conversation.first_look = asyncio.ensure_future(self.backend.search(request))
            conversation.detected = asyncio.ensure_future(
                self.backend.detect(to_jpeg(photo, DETECT_SIDE))
            )
        self._conversations[conversation.id] = conversation
        conversation.log(
            {
                "type": "open",
                "photo": photo is not None,
                "words": conversation.words,
                "position": position is not None,
                "oneshot": oneshot,
                "filters": not conversation.filters.empty,
                "model": self.model.model,
            }
        )
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
        start = time.monotonic()
        c.log(
            {
                "type": "visitor",
                "kind": given.kind.value,
                "text": given.text.strip()[:MAX_WORDS],
                "region": given.region is not None,
                "position": given.position is not None,
            }
        )
        try:
            await self._take(c, given)
            waiting = await self._loop(c)
        except Exception:
            logger.exception("chat %s: turn failed", c.id[:8])
            c.emit({"type": "error", "text": chat_prompts.TURN_FAILED})
            c.log({"type": "error"})
        c.emit({"type": "end", "waiting": waiting})
        c.log(
            {
                "type": "end",
                "waiting": waiting,
                "searches": c.method.turn.searches,
                "lock": c.method.turn.lock,
                "best": list(c.method.best),
                "spent_usd": round(c.spent_usd, 6),
                "calls": c.calls,
                "ms": _ms(start),
            }
        )
        await self.log.flush()

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
            c.messages.append(
                user_message(
                    text_part(chat_prompts.VISITOR_WRITES.format(text=text, shared=shared))
                )
            )
            return
        if pending.kind is Wait.REGION and given.kind is Visit.REGION:
            if given.region is None:
                c.messages[pending.message]["content"] = chat_prompts.FRAMED_NOTHING
                return
            c.region_bytes = given.region
            c.messages[pending.message]["content"] = chat_prompts.FRAMED
            c.messages.append(
                user_message(
                    text_part(chat_prompts.FRAMED_PART), image_ref("region", len(c.thumbs))
                )
            )
            return
        answer = text or chat_prompts.NO_ANSWER
        c.messages[pending.message]["content"] = chat_prompts.VISITOR_ANSWERS.format(
            answer=answer, shared=shared
        )

    async def _opening(self, c: Conversation, goal: str) -> Message:
        parts: list[JSON] = [text_part(chat_prompts.GOAL.format(goal=goal or "?"))]
        if c.oneshot:
            parts.append(text_part(ONESHOT_NOTE))
        if not c.filters.empty:
            parts.append(text_part(chat_prompts.filters_note(c.filters)))
        if c.photo_bytes is None:
            parts.append(text_part(chat_prompts.NO_PHOTO.format(words=c.words)))
            return user_message(*parts)
        parts += [text_part(chat_prompts.PHOTO), image_ref("photo", 0)]
        if c.words:
            parts.append(text_part(chat_prompts.WORDS_WITH_PHOTO.format(words=c.words)))
        if c.position_given is not None:
            parts.append(text_part(chat_prompts.PHOTO_POSITION))
        parts.append(text_part(await self._first_look(c)))
        parts.append(text_part(chat_prompts.outlines_note(await c.outlines())))
        return user_message(*parts)

    async def _first_look(self, c: Conversation) -> str:
        """The site's usual search, started when the conversation opened."""
        assert c.first_look is not None
        c.steps += 1
        step = c.steps
        label = chat_prompts.FIRST_LOOK_LABEL
        c.emit({"type": "step", "id": step, "state": "running", "label": label})
        start = time.monotonic()
        try:
            found = await asyncio.wait_for(asyncio.shield(c.first_look), FIRST_LOOK_WAIT_S)
        except (TimeoutError, ValueError, RuntimeError) as err:
            logger.warning("chat %s: first look failed: %s", c.id[:8], err)
            c.emit(step_event(step, label, "pas de résultat", (), error=True))
            c.log({"type": "first_look", "error": True, "ms": _ms(start)})
            return chat_prompts.FIRST_LOOK_FAILED
        cards = found.cards[:FIRST_LOOK]
        c.method.first_look(
            [k.entity_id for k in cards],
            (k.entity_id for k in cards if k.level is Level.VERY_CLOSE),
        )
        c.log(
            {
                "type": "first_look",
                "hits": [k.entity_id for k in found.cards],
                "levels": [k.level.value for k in cards],
                "caption": found.caption,
                "ms": _ms(start),
            }
        )
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
        c.emit(step_event(step, label, f"{len(found.cards)} résultats", thumbs))
        lines = [
            f"{i}. {k.entity_id} | {k.collection} | {k.title} | {k.place} | {k.level_label}"
            for i, k in enumerate(cards, start=1)
        ]
        head = chat_prompts.FIRST_LOOK_HEAD
        if found.caption:
            head += "\n" + chat_prompts.FIRST_LOOK_CAPTION.format(caption=found.caption)
        if not lines:
            return head + "\nnothing found"
        sure = c.method.sure
        sure_line = chat_prompts.FIRST_LOOK_SURE.format(ids=", ".join(sorted(sure))) if sure else ""
        look = chat_prompts.FIRST_LOOK_LOOK.format(ids=", ".join(c.method.must_view))
        return f"{head}\n" + "\n".join(lines) + f"\n{sure_line}\n{look}"

    async def _loop(self, c: Conversation) -> str:
        """Model calls and their tools until the model shows results, asks, or stops."""
        tools = tools_for(photo=c.photo_bytes is not None, ask=not c.oneshot)
        c.method.new_turn()
        schemas = [t.schema() for t in tools]
        offered = {t.name.value for t in tools}
        told_last = False
        told_lock = False
        nudged = False
        for _ in range(MAX_MODEL_CALLS):
            stop = await self._over_limit(c)
            if stop is not None:
                c.emit({"type": "limit", "text": stop})
                c.log({"type": "limit", "text": stop})
                return "limit"
            out_of_calls = c.method.out_of_calls
            if out_of_calls and not told_last:
                c.messages.append(
                    user_message(
                        text_part(chat_prompts.LAST_CALL.format(calls=chat_method.MAX_TOOL_CALLS))
                    )
                )
                told_last = True
            lock = c.method.turn.lock
            if lock is not None and not told_lock and not told_last:
                c.messages.append(user_message(text_part(LOCKED_NOTE.format(reason=lock))))
                told_lock = True
            body: dict[str, JSON] = {
                "messages": self._render(c),
                "tools": schemas,
                "tool_choice": "required" if out_of_calls or told_lock else "auto",
                "max_tokens": MAX_TOKENS,
            }
            c.emit({"type": "status", "text": chat_prompts.THINKING})
            start = time.monotonic()
            try:
                answer = await self.model.complete(body)
            except (OpenRouterError, MistralError) as err:
                logger.warning("chat %s: model call failed: %s", c.id[:8], err)
                c.emit({"type": "error", "text": chat_prompts.MODEL_DOWN})
                c.log({"type": "model", "error": str(err)[:200], "ms": _ms(start)})
                return "message"
            await self._paid(c, answer.cost_usd)
            c.log(
                {
                    "type": "model",
                    "call": c.calls,
                    "text": answer.text,
                    "thinking": answer.thinking,
                    "tool_calls": [
                        {"name": t.name, "args": redact(arguments_of(t.arguments))}
                        for t in answer.tool_calls
                    ],
                    "cost_usd": answer.cost_usd,
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                    "ms": _ms(start),
                }
            )
            said = answer.text.strip()
            if said and not summarised(answer.tool_calls):
                c.emit({"type": "say", "text": said})
            if not answer.tool_calls:
                # Mistral refuses an assistant message with neither words nor calls.
                c.messages.extend([{"role": "assistant", "content": answer.text}] if said else [])
                if nudged:
                    return "message"
                # Mistral sometimes says what it will do next and stops: once a turn, it is
                # asked to do it.
                c.messages.append(user_message(text_part(chat_prompts.NUDGE)))
                nudged = True
                continue
            ids = [call_id_of(call, c.calls, n) for n, call in enumerate(answer.tool_calls)]
            c.messages.append(assistant_message(answer.text, answer.tool_calls, ids))
            waiting, thumbs = await self._calls(c, answer.tool_calls, ids, offered)
            c.messages.extend(shown_images(thumbs, c.query_images))
            if waiting is not None:
                return waiting
        c.emit({"type": "say", "text": chat_prompts.GAVE_UP})
        return "message"

    async def _calls(
        self,
        c: Conversation,
        calls: Sequence[ToolCall],
        ids: Sequence[str],
        offered: set[str],
    ) -> tuple[str | None, list[tuple[str, str]]]:
        """Run one reply's tool calls."""
        method = c.method
        waiting: str | None = None
        thumbs: list[tuple[str, str]] = []
        for call, call_id in zip(calls, ids, strict=True):
            args = arguments_of(call.arguments)
            refused = method.refusal(
                call.name, args, offered=offered, waiting=waiting is not None, seen=c.seen
            )
            if refused is not None:
                c.messages.append(tool_message(call_id, call.name, refused))
                c.log({"type": "tool", "name": call.name, "args": redact(args), "refused": refused})
                logger.info("chat %s: %s refused: %s", c.id[:8], call.name, refused[:60])
                continue
            counted = call.name in COUNTED
            c.steps += 1
            step = c.steps
            why = why_of(args)
            if counted:
                running: Event = {
                    "type": "step",
                    "id": step,
                    "state": "running",
                    "label": running_label(call.name, args) or chat_prompts.RUNNING[call.name],
                }
                if why:
                    running["why"] = why
                c.emit(running)
            best_before = tuple(method.best)
            start = time.monotonic()
            outcome = await run_tool(c, call.name, call.arguments)
            method.ran(call.name, outcome, best_before)
            logger.info(
                "chat %s step %d: %s%s",
                c.id[:8],
                step,
                call.name,
                " (error)" if outcome.error else "",
            )
            c.log(
                {
                    "type": "tool",
                    "step": step,
                    "name": call.name,
                    "args": redact(args),
                    "why": why,
                    "hits": list(outcome.hit_ids),
                    "error": outcome.error,
                    "text": outcome.text[:300] if outcome.error else "",
                    "found": None if outcome.found is None else outcome.found.value,
                    "summary": outcome.summary,
                    "picks": [p.entity_id for p in outcome.picks or ()],
                    "ms": _ms(start),
                }
            )
            ran = not outcome.error
            if ran and (counted or call.name == ToolName.VIEW_RESULTS.value):
                outcome = with_note(outcome, method.best_line(c.title_of))
            waiting = self._outcome(c, outcome, step, call_id, call.name, why=why)
            thumbs += [(t.label, t.image_id) for t in outcome.thumbnails]
        return waiting, thumbs

    def _outcome(
        self,
        c: Conversation,
        outcome: ToolOutcome,
        step: int,
        call_id: str,
        name: str,
        *,
        why: str = "",
    ) -> str | None:
        """Put one tool's outcome in the messages and the events; what to wait for, if any."""
        if outcome.wait is not None:
            c.messages.append(tool_message(call_id, name, chat_prompts.WAITING))
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
        c.messages.append(tool_message(call_id, name, outcome.text))
        if outcome.picks is not None:
            cards: list[JSON] = []
            for pick in outcome.picks[:MAX_RESULTS]:
                known = c.known[pick.entity_id]
                cards.append(card_json(known, pick.reason))
            if outcome.summary:
                c.emit({"type": "say", "text": outcome.summary})
            found = None if outcome.found is None else outcome.found.value
            c.emit({"type": "results", "cards": cards, "found": found})
            return "message"
        if outcome.step is not None:
            s = outcome.step
            thumbs: list[JSON] = [
                {"entity_id": t.entity_id, "title": t.title, "image_url": t.image_url}
                for t in s.thumbs
            ]
            event = step_event(step, s.label, s.summary, thumbs, error=s.error)
            if why:
                event["why"] = why
            c.emit(event)
        return None

    # --- money ---------------------------------------------------------------------------------

    async def _over_limit(self, c: Conversation) -> str | None:
        """Why the next call may not be made, or None. It is refused when it could take the
        conversation or the day past its cap, guessing its cost from the last call's."""
        guess = c.last_call_usd
        if c.spent_usd + guess > self.limits.conversation_usd:
            logger.info("chat %s: conversation cap reached at $%.4f", c.id[:8], c.spent_usd)
            return chat_prompts.CONVERSATION_CAP.format(cap=_dollars(self.limits.conversation_usd))
        if await self.spend.spent(self._today()) + guess > self.limits.day_usd:
            logger.info("chat %s: day cap reached", c.id[:8])
            return chat_prompts.DAY_CAP
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
            calls=chat_method.MAX_TOOL_CALLS,
            views=chat_method.MAX_VIEWS,
            searches=chat_method.MAX_SEARCHES,
            stale=chat_method.MAX_STALE,
        )

    def _render(self, c: Conversation) -> list[JSON]:
        """The messages with images as data URLs; each thumbnail is checked as publishable
        once more before it leaves."""
        publishable = self.thumbs.publishable
        return render(
            c.messages,
            photo=c.photo_bytes,
            region=c.region_bytes,
            thumbs=c.thumbs,
            query_images=c.query_images,
            check=lambda image_id: publishable.check(image_id, "the model"),
        )


def _ms(start: float) -> float:
    return round(1000 * (time.monotonic() - start), 1)


def _dollars(usd: float) -> str:
    return f"{usd:.2f} $".replace(".", ",")
