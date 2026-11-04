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
from PIL import Image as PILImage

from miroir.images.fetch import to_jpeg
from miroir.jobs.openrouter import JSON, ChatAnswer, ToolCall
from miroir.outlines import DETECT_SIDE, Box, Outlines
from miroir.web import chat_method, chat_prompts
from miroir.web.catalog import Catalog, CatalogRecord, Filters
from miroir.web.chat_method import COUNTED, FINISHING, MAX_REFUSED, Method, early_question
from miroir.web.chat_model import ChatModel, MistralModel, ModelError, OpenRouterModel, chat_model
from miroir.web.chat_prompts import BRIEF, LOCKED_NOTE
from miroir.web.chat_render import (
    FIRST_LOOK,
    KEEP_QUERY_IMAGES,
    PHOTO_SIDE,
    CardView,
    Event,
    Message,
    assistant_message,
    call_id_of,
    card_json,
    card_of_hit,
    data_url,
    image_ref,
    render,
    sent_thumbs,
    shown_images,
    step_event,
    summarised,
    text_part,
    tool_message,
    user_message,
    with_note,
)
from miroir.web.chat_spend import Budget, Cap, FileSpend, Hold, Limits, MemorySpend, SpendStore
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
    "BusyError",
    "CardView",
    "ChatModel",
    "ChatService",
    "Conversation",
    "FileSpend",
    "Limits",
    "MemorySpend",
    "MistralModel",
    "ModelError",
    "OpenRouterModel",
    "SpendStore",
    "ThumbSource",
    "ToolBackend",
    "Visit",
    "VisitorInput",
    "chat_model",
]

logger = logging.getLogger(__name__)

MAX_TOKENS = 800
#: Model calls per visitor message, all told.
MAX_MODEL_CALLS = 25
#: Seconds the model may work on one visitor message, its first look apart.
TURN_S = 300.0
#: First-look results the model is told about.
FIRST_LOOK_TOLD = 10
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


class ToolBackend(Protocol):
    """What the agent needs of the search, one of its two seams (the other is `ChatModel`).

    The site serves it on the live engine (`miroir.web.app.SearchBackend`); the offline
    evaluation on a bundle's stored vectors (`miroir.eval.offline.OfflineBackend`). The rest
    of what the tools use comes with `ChatService`: records to read (`Catalog`), and the one
    check of what may go to the model (`ThumbSource`, `miroir.web.publishable`)."""

    async def search(self, request: SearchRequest) -> SearchResponse:
        """The site's usual search: the first look."""
        ...

    async def tool_search(self, query: ToolQuery) -> ToolHits:
        """Image search, text search, `more_like`, and the images of entities to view
        (`miroir.web.toolsearch`); unknown filter values come back in `ToolHits.unknown`."""
        ...

    async def detect(self, jpeg: bytes) -> Outlines:
        """The works in a photo (F4), as the photo view outlines them: what a transform may
        crop to."""
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
        except (httpx.HTTPError, OSError, ValueError, PILImage.DecompressionBombError) as err:
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


class _Reply:
    """One model reply's messages: its tool calls, then a reply to each. They join the
    conversation only once every call has its reply."""

    __slots__ = ("messages", "refused", "thumbs", "wait", "waiting")

    def __init__(self, assistant: Message) -> None:
        self.messages: list[Message] = [assistant]
        self.thumbs: list[tuple[str, str]] = []
        """(label, image id or query image key) to show the model after the replies."""
        self.wait: tuple[Wait, int] | None = None
        """What a question waits for, and its tool reply's index in `messages`."""
        self.waiting: str | None = None
        self.refused = 0
        """Calls the method refused."""


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
        filters: Filters | None = None,
    ) -> None:
        self.service = service
        self.id = secrets.token_urlsafe(16)
        self.session = session
        self.photo_bytes = photo
        self.words = words
        self.position_given = position
        self.filters = filters or Filters()
        """The filters the visitor set on the page."""
        self.touched = now
        self.messages: list[Message] = []
        self.events: list[Event] = []
        self.changed = asyncio.Event()
        self.first_look: asyncio.Future[SearchResponse] | None = None
        self.detected: asyncio.Future[Outlines] | None = None
        """The detector's outlines of the photo, started with the first look."""
        self.query_images: dict[str, JSON] = {}
        """Transformed photos the model searched, kept for it to look at, by key."""
        self.query_count = 0
        self.stream_id = 0
        """The number of the one event stream open: a new one ends the one before."""
        self.task: asyncio.Future[None] | None = None
        self.running = False
        """A turn is under way: set when it starts, cleared just before its end event."""
        self.pending: _Pending | None = None
        self.started = False
        self.known: dict[str, CardView] = {}
        """Every entity the model has seen in a result: what show_results may show."""
        self.signatures: list[tuple[Signature, int]] = []
        self.regions: list[bytes] = []
        """The parts of the photo the visitor framed, in order; searches take the last."""
        self.thumbs: dict[str, JSON] = {}
        """Thumbnails fetched for the model, by image id (each one publishable), as data URLs.
        The images below are kept as data URLs too, each encoded once."""
        self.photo_image: JSON | None = None
        self.region_images: list[JSON] = []
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
        return self.regions[-1] if self.regions else None

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
                self.thumbs[image.image_id] = data_url(data)
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
        except Exception as err:  # no outlines: the model crops by box or cell
            logger.warning("chat %s: outlines failed: %r", self.id[:8], err)
            return ()
        return found.boxes[:OUTLINES_LISTED]

    def keep_query_image(self, jpeg: bytes) -> str:
        self.query_count += 1
        key = f"query-{self.query_count}"
        self.query_images[key] = data_url(jpeg)
        return key

    def forget_images(self) -> None:
        """Drop the images the model is no longer sent: older thumbnails and transformed
        photos become a note in the messages (`miroir.web.chat_render.render`)."""
        sent = sent_thumbs(self.messages)
        for image_id in [i for i in self.thumbs if i not in sent]:
            del self.thumbs[image_id]
        for key in list(self.query_images)[:-KEEP_QUERY_IMAGES]:
            del self.query_images[key]

    def shortlist(self) -> list[JSON]:
        """The best-so-far as the page shows it: id, title and image of each."""
        cards = [self.known[e] for e in self.method.best if e in self.known]
        return [
            {"entity_id": k.entity_id, "title": k.title, "image_url": k.image_url} for k in cards
        ]

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
        return None

    # --- events --------------------------------------------------------------------------------

    def emit(self, event: Event) -> None:
        self.events.append(event)
        self.wake()

    def wake(self) -> None:
        """Wake the open stream: an event came, or a new stream replaces it."""
        done, self.changed = self.changed, asyncio.Event()
        done.set()

    def log(self, line: Mapping[str, JSON]) -> None:
        self.service.log.write(self.day, self.id, line)


class ChatService:
    """Every open conversation, and the means they share."""

    def __init__(
        self,
        *,
        backend: ToolBackend,
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
        self._today = today or (lambda: datetime.now(UTC).date().isoformat())
        self.budget = Budget(spend, limits, self._today)
        self.index_size = index_size
        """(images, entities) the index holds, for the brief."""
        self.catalog = catalog
        """Record text for read_record; None reads nothing."""
        self._clock = clock
        self._conversations: dict[str, Conversation] = {}
        self._background: set[asyncio.Future[None]] = set()
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
        filters: Filters | None = None,
    ) -> Conversation:
        """A new conversation. With a photo, the site's usual search starts at once.
        `filters`: those the visitor set, which the first look applies and the model is told
        about."""
        self._forget_old()
        conversation = Conversation(
            self,
            session=session,
            photo=photo,
            words=words.strip()[:MAX_WORDS],
            position=position,
            now=self._clock(),
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
            conversation.detected = asyncio.ensure_future(self._detect(photo))
        self._conversations[conversation.id] = conversation
        conversation.log(
            {
                "type": "open",
                "photo": photo is not None,
                "words": conversation.words,
                "position": position is not None,
                "filters": not conversation.filters.empty,
                "model": self.model.model,
            }
        )
        logger.info(
            "chat %s opened: photo=%s words=%d", conversation.id[:8], photo is not None, len(words)
        )
        return conversation

    async def _detect(self, photo: bytes) -> Outlines:
        return await self.backend.detect(await asyncio.to_thread(to_jpeg, photo, DETECT_SIDE))

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
        conversation.running = True
        conversation.task = asyncio.ensure_future(self._turn(conversation, given))
        return after

    async def stream(self, conversation: Conversation, after: int) -> AsyncIterator[Event]:
        """Events from number `after` on, until the turn ends or `STREAM_S` pass. One stream
        per conversation: a new one ends the one before, which says it was replaced."""
        deadline = self._clock() + STREAM_S
        n = max(after, 0)
        conversation.stream_id += 1
        mine = conversation.stream_id
        conversation.wake()
        while True:
            if conversation.stream_id != mine:
                yield {"type": "replaced"}
                return
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
            async with asyncio.timeout(TURN_S):
                waiting = await self._loop(c)
        except TimeoutError:
            logger.warning("chat %s: turn out of time", c.id[:8])
            c.emit({"type": "error", "text": chat_prompts.OUT_OF_TIME})
            c.log({"type": "error", "text": "out of time"})
        except Exception:
            logger.exception("chat %s: turn failed", c.id[:8])
            c.emit({"type": "error", "text": chat_prompts.TURN_FAILED})
            c.log({"type": "error"})
        finally:
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
            # Not running before the end is told: the visitor may write as soon as they read
            # it, and the stream closes with it. The log is made durable in the background.
            c.running = False
            c.emit({"type": "end", "waiting": waiting})
            self._later(self.log.flush())

    def _later(self, work: Awaitable[None]) -> None:
        """Run `work` in the background, kept until it is done."""
        task = asyncio.ensure_future(work)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _take(self, c: Conversation, given: VisitorInput) -> None:
        """Add what the visitor sent to the messages."""
        if given.position is not None:
            c.position_given = given.position
        text = given.text.strip()[:MAX_WORDS]
        if not c.started:
            opening = await self._opening(c, text)
            c.messages = [{"role": "system", "content": self._brief()}, opening]
            c.started = True  # only now: a failed opening is built again with the next message
            return
        # Never the coordinates: near "visitor" is resolved here, and the model could repeat
        # them in words the visitor reads and the log keeps.
        shared = "" if given.position is None else chat_prompts.POSITION_SHARED
        pending, c.pending = c.pending, None
        if pending is None:
            c.messages.append(
                user_message(
                    text_part(chat_prompts.VISITOR_WRITES.format(text=text, shared=shared))
                )
            )
            return
        # A framed part answers a request to frame, or a question that offered framing.
        if given.kind is Visit.REGION:
            if given.region is None:
                c.messages[pending.message]["content"] = chat_prompts.FRAMED_NOTHING
                return
            c.region_images.append(await _for_model(given.region))
            c.regions.append(given.region)
            c.messages[pending.message]["content"] = chat_prompts.FRAMED
            c.messages.append(
                user_message(
                    text_part(chat_prompts.FRAMED_PART), image_ref("region", len(c.regions) - 1)
                )
            )
            return
        answer = text or chat_prompts.NO_ANSWER
        c.messages[pending.message]["content"] = chat_prompts.VISITOR_ANSWERS.format(
            answer=answer, shared=shared
        )

    async def _opening(self, c: Conversation, goal: str) -> Message:
        parts: list[JSON] = [text_part(chat_prompts.GOAL.format(goal=goal or "?"))]
        if not c.filters.empty:
            parts.append(text_part(chat_prompts.filters_note(c.filters)))
        if c.photo_bytes is None:
            parts.append(text_part(chat_prompts.NO_PHOTO.format(words=c.words)))
            return user_message(*parts)
        if c.photo_image is None:
            c.photo_image = await _for_model(c.photo_bytes)
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
        c.emit({"type": "step", "id": step, "tool": FIRST_LOOK, "state": "running", "label": label})
        start = time.monotonic()
        try:
            found = await asyncio.wait_for(asyncio.shield(c.first_look), FIRST_LOOK_WAIT_S)
        except Exception as err:  # the model can search on its own: the turn goes on
            logger.warning("chat %s: first look failed: %r", c.id[:8], err)
            c.emit(step_event(step, FIRST_LOOK, label, "pas de résultat", (), error=True))
            c.log({"type": "first_look", "error": True, "ms": _ms(start)})
            return chat_prompts.FIRST_LOOK_FAILED
        cards = found.cards[:FIRST_LOOK_TOLD]
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
        done = step_event(step, FIRST_LOOK, label, f"{len(found.cards)} résultats", thumbs)
        c.emit({**done, "shortlist": c.shortlist()})
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
        ask = early_question(
            building=found.guessed_type == "building",
            position=c.position_given is not None,
            caption_place=found.caption_place is not None,
            sure=bool(sure),
        )
        return f"{head}\n" + "\n".join(lines) + f"\n{sure_line}\n{look}\n{ask}".rstrip()

    async def _loop(self, c: Conversation) -> str:
        """Model calls and their tools until the model shows results, asks, or stops."""
        tools = tools_for(photo=c.photo_bytes is not None)
        c.method.new_turn()
        schemas = [t.schema() for t in tools]
        # Locked or out of searches, the model is offered only what still runs.
        finishing = [t.schema() for t in tools if t.name in FINISHING]
        offered = {t.name.value for t in tools}
        told_last = False
        told_lock = False
        nudged = False
        refused = 0
        for _ in range(MAX_MODEL_CALLS):
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
                "tools": finishing if c.method.finishing else schemas,
                "tool_choice": "required" if out_of_calls or told_lock else "auto",
                "max_tokens": MAX_TOKENS,
            }
            answer = await self._ask(c, body)
            if isinstance(answer, str):
                return answer
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
            reply = _Reply(assistant_message(answer.text, answer.tool_calls, ids))
            await self._calls(c, answer.tool_calls, ids, offered, reply)
            reply.messages += shown_images(reply.thumbs, c.query_images)
            # Only now, every call answered: the history the model is sent stays valid.
            if reply.wait is not None:
                kind, at = reply.wait
                c.pending = _Pending(kind, len(c.messages) + at)
            c.messages.extend(reply.messages)
            c.forget_images()
            if reply.waiting is not None:
                return reply.waiting
            refused = refused + 1 if reply.refused == len(ids) else 0
            if refused >= MAX_REFUSED:  # each refused reply is a paid call that did nothing
                logger.info("chat %s: %d replies refused in a row", c.id[:8], refused)
                break
        c.emit({"type": "say", "text": chat_prompts.GAVE_UP})
        return "message"

    async def _ask(self, c: Conversation, body: Mapping[str, JSON]) -> ChatAnswer | str:
        """One paid model call, within the caps; what to wait for when none is made."""
        # Refused when it could take the conversation or the day past its cap, guessing its
        # cost from the last call's (the next one sends a little more).
        hold = await self.budget.hold(c.spent_usd, c.last_call_usd)
        if isinstance(hold, Cap):
            stop = self._cap_text(c, hold)
            c.emit({"type": "limit", "text": stop})
            c.log({"type": "limit", "text": stop})
            return "limit"
        c.emit({"type": "status", "text": chat_prompts.THINKING})
        start = time.monotonic()
        try:
            answer = await self.model.complete(body)
        except ModelError as err:
            logger.warning("chat %s: model call failed: %s", c.id[:8], err)
            if err.maybe_billed:  # it may have run: count what it would have cost
                await self._paid(c, hold, hold.usd)
            else:
                self.budget.release(hold)
            c.emit({"type": "error", "text": chat_prompts.MODEL_DOWN})
            c.log({"type": "model", "error": str(err)[:200], "ms": _ms(start)})
            return "message"
        except BaseException:  # the turn's deadline: the call may run on, and be paid for
            await self._paid(c, hold, hold.usd)
            raise
        await self._paid(c, hold, answer.cost_usd)
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
        return answer

    async def _calls(
        self,
        c: Conversation,
        calls: Sequence[ToolCall],
        ids: Sequence[str],
        offered: set[str],
        reply: _Reply,
    ) -> None:
        """Run one reply's tool calls, each answered in `reply`."""
        method = c.method
        for call, call_id in zip(calls, ids, strict=True):
            args = arguments_of(call.arguments)
            refused = method.refusal(
                call.name, args, offered=offered, waiting=reply.waiting is not None, seen=c.seen
            )
            if refused is not None:
                reply.refused += 1
                reply.messages.append(tool_message(call_id, call.name, refused))
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
                    "tool": call.name,
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
            if outcome.signature is not None and not outcome.error:
                c.signatures.append((outcome.signature, step))
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
            ask = method.ask_note() if ran and counted else None
            if ask is not None:
                outcome = with_note(outcome, ask)
            reply.waiting = self._outcome(
                c, reply, outcome, step=step, call_id=call_id, name=call.name, why=why
            )
            reply.thumbs += [(t.label, t.image_id) for t in outcome.thumbnails]

    def _outcome(
        self,
        c: Conversation,
        reply: _Reply,
        outcome: ToolOutcome,
        *,
        step: int,
        call_id: str,
        name: str,
        why: str = "",
    ) -> str | None:
        """Put one tool's outcome in the reply and the events; what to wait for, if any."""
        if outcome.wait is not None:
            reply.wait = (outcome.wait, len(reply.messages))
            reply.messages.append(tool_message(call_id, name, chat_prompts.WAITING))
            if outcome.question is not None:
                q = outcome.question
                c.emit(
                    {
                        "type": "question",
                        "text": q.text,
                        "choices": list(q.choices),
                        "about": q.about.value,
                        "position": q.offer_position,
                        "region": q.offer_region,
                    }
                )
                return "answer"
            c.emit({"type": "region", "text": outcome.region_reason})
            return "region"
        reply.messages.append(tool_message(call_id, name, outcome.text))
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
            event = step_event(step, name, s.label, s.summary, thumbs, error=s.error)
            if why:
                event["why"] = why
            if not s.error and (name in COUNTED or name == ToolName.VIEW_RESULTS.value):
                event["shortlist"] = c.shortlist()
            c.emit(event)
        return None

    # --- money ---------------------------------------------------------------------------------

    def _cap_text(self, c: Conversation, cap: Cap) -> str:
        if cap is Cap.CONVERSATION:
            logger.info("chat %s: conversation cap reached at $%.4f", c.id[:8], c.spent_usd)
            return chat_prompts.CONVERSATION_CAP.format(cap=_dollars(self.limits.conversation_usd))
        logger.info("chat %s: day cap reached", c.id[:8])
        return chat_prompts.DAY_CAP

    async def _paid(self, c: Conversation, hold: Hold, usd: float) -> None:
        c.calls += 1
        c.spent_usd += usd
        if usd > 0:  # the next call sends a little more: guess it costs what this one did
            c.last_call_usd = usd
        day = await self.budget.settle(hold, usd)
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
            questions=chat_method.MAX_QUESTIONS,
        )

    def _render(self, c: Conversation) -> list[JSON]:
        """The messages with images as data URLs; each thumbnail is checked as publishable
        once more before it leaves."""
        publishable = self.thumbs.publishable
        return render(
            c.messages,
            photo=c.photo_image,
            regions=c.region_images,
            thumbs=c.thumbs,
            query_images=c.query_images,
            check=lambda image_id: publishable.check(image_id, "the model"),
        )


async def _for_model(image: bytes) -> JSON:
    """The visitor's photo or framed part as the model sees it, encoded off the event loop."""
    return data_url(await asyncio.to_thread(to_jpeg, image, PHOTO_SIDE))


def _ms(start: float) -> float:
    return round(1000 * (time.monotonic() - start), 1)


def _dollars(usd: float) -> str:
    return f"{usd:.2f} $".replace(".", ",")
