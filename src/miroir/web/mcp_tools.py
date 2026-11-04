"""What the MCP connector's tools do (`miroir.web.mcp` serves them): search, fetch, delegate.

- `search`: the site's simple search (photo or words, filters, position), the same engine and
  card levels as `POST /api/search`.
- `fetch`: one building or object in full, as its record page shows it.
- `delegate_search`: a turn of the site's own agent (`miroir.web.chat`, step H5), run on the
  server; its picks with a reason each, or the question it asks.

Each caller is one connector token, by name: its own rate limits, its own agent conversations
(the chat service's session is `mcp:<name>`). The agent keeps the site's spending caps per
conversation and per day, shared with the site's visitors.

Images given to callers are publishable ones only (`miroir.web.publishable`, technical spec
§9): a link to an image goes out of our hands, to the calling model's service.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from miroir.web import auth, facets
from miroir.web.app import SEARCH_WAIT_S, Site, entity_page
from miroir.web.catalog import (
    CARD_SIDE,
    COLLECTION_LABELS,
    Catalog,
    CatalogRecord,
    FacetValue,
    Filters,
    Source,
    display_url,
)
from miroir.web.chat import BusyError, ChatService, Conversation, Visit, VisitorInput
from miroir.web.chat_tools import Wait
from miroir.web.engine import MAX_TEXT, Card, SearchRequest
from miroir.web.levels import LEVEL_LABELS, Level
from miroir.web.mcp_photo import PhotoError, PhotoFetcher, checked_photo, decode_base64
from miroir.web.nearby import Position
from miroir.web.publishable import Publishable

logger = logging.getLogger(__name__)

#: Results a search may return.
MAX_TOP_K = 20
#: Searches and fetches per caller per minute; agent turns (they cost money) per minute.
CALLS_PER_MINUTE = 30
AGENT_TURNS_PER_MINUTE = 4
#: How long `delegate_search` waits for the agent by default, and at most. An agent turn
#: takes 20 s to 2 min; past the wait the caller gets "running" and calls again to collect.
AGENT_WAIT_S = 50
MAX_AGENT_WAIT_S = 110
#: Images listed by `fetch`; steps reported by `delegate_search`.
FETCH_IMAGES = 12
STEPS_SHOWN = 20

CollectionKey = Literal["joconde", "memoire", "rose-valland", "merimee", "palissy", "museo"]
PeriodName = Literal[
    "Préhistoire",
    "Antiquité",
    "Moyen Âge",
    "16e siècle",
    "17e siècle",
    "18e siècle",
    "19e siècle",
    "20e siècle",
    "21e siècle",
]


class ToolCallError(Exception):
    """A call that cannot run as asked; the message tells the calling model what to change."""


# --- what the tools return ------------------------------------------------------------------

_FROZEN = ConfigDict(frozen=True)


class RecordHit(BaseModel):
    """One building or object (an entity: a record, or the records of one building)."""

    model_config = _FROZEN

    id: str = Field(description="MIROIR id of the building or object: pass it to `fetch`.")
    title: str
    url: str = Field(description="Its record on POP (pop.culture.gouv.fr): cite this link.")
    collection: str = Field(description="POP base: Joconde, Mémoire, Mérimée, Palissy…")
    place: str = Field(description="Commune and département, when known.")
    period: list[str]
    authors: list[str]
    level: str = Field(
        description="tres-proche (most likely the very record), proche (maybe it: check), "
        "ressemblance (looks alike). Only the first result can be above ressemblance."
    )
    level_label: str = Field(description="The level as the site shows it, in French.")
    thumbnail_url: str | None = Field(
        description="A POP image of it that may be shown, or null when none may."
    )
    photo_count: int = Field(description="POP images of it.")
    reason: str | None = Field(
        default=None, description="delegate_search only: why the agent picked it (French)."
    )


class SearchResult(BaseModel):
    model_config = _FROZEN

    results: list[RecordHit] = Field(description="Closest first.")
    mode: str = Field(description='"photo", "text" or "photo+text".')
    notes: list[str] = Field(description="What the search noticed, for the reader.")


class LinkedRecord(BaseModel):
    model_config = _FROZEN

    ref: str
    title: str
    url: str


class RecordMetadata(BaseModel):
    model_config = _FROZEN

    collection: str
    place: str
    facts: dict[str, str] = Field(description="Dénomination, domaine, auteurs, période, lieu…")
    records: list[LinkedRecord] = Field(description="The POP records of this building or object.")
    images: list[str] = Field(description="POP images of it that may be shown.")
    photo_count: int


class Document(BaseModel):
    """A record in full, in the shape ChatGPT's deep research reads (`fetch`)."""

    model_config = _FROZEN

    id: str
    title: str
    text: str = Field(description="Description and key facts, in French as POP writes them.")
    url: str
    metadata: RecordMetadata


AgentStatus = Literal["done", "question", "frame", "running", "stopped", "error"]
#: How a finished turn's "end" event (what the page waits for) reads as a status.
_ENDS: dict[str, AgentStatus] = {"answer": "question", "region": "frame", "limit": "stopped"}


class AgentAnswer(BaseModel):
    model_config = _FROZEN

    conversation_id: str = Field(description="Pass it back to answer or to go on.")
    status: AgentStatus = Field(
        description="done: results below. question: relay `question` to the user and call "
        "again with their answer as `reply`. frame: the agent wants the part of the photo "
        "that matters, as `region_image_url` (or a `reply` in words). running: the agent is "
        "still working; call again with only conversation_id. stopped: a spending cap was "
        "reached. error: the agent failed; use `search` instead."
    )
    summary: str = Field(description="What the agent says (French), to pass on to the user.")
    found: str | None = Field(
        description="exact (the very record), close (the nearest ones), none (not in POP)."
    )
    results: list[RecordHit] = Field(description="The agent's picks, best first.")
    question: str | None
    choices: list[str] = Field(description="Answers the agent suggests for its question.")
    steps: list[str] = Field(description="What the agent did, in order (French).")


# --- the tools ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilterArgs:
    collections: Sequence[str] = ()
    region: str | None = None
    departement: str | None = None
    domain: str | None = None
    period: str | None = None


class McpTools:
    """The tools' work over the site's search, catalog and agent. `caller` names the token."""

    def __init__(self, *, site: Site, publishable: Publishable, photos: PhotoFetcher) -> None:
        self.site = site
        self.publishable = publishable
        self.photos = photos
        self.call_limit = auth.RateLimit(CALLS_PER_MINUTE)
        self.agent_limit = auth.RateLimit(AGENT_TURNS_PER_MINUTE)
        self._turn_start: dict[str, int] = {}
        """Where each conversation's latest turn starts in its events."""

    @property
    def catalog(self) -> Catalog:
        return self.site.catalog

    @property
    def agent(self) -> ChatService | None:
        return self.site.chat

    # --- search ----------------------------------------------------------------------------

    async def search(
        self,
        caller: str,
        *,
        query: str = "",
        image_url: str | None = None,
        image_base64: str | None = None,
        filters: FilterArgs | None = None,
        position: Position | None = None,
        top_k: int = 10,
    ) -> SearchResult:
        self._allow(caller, self.call_limit)
        photo = await self.photo(image_url, image_base64)
        if photo is None and not query.strip():
            msg = "give a query (words), an image_url, or both"
            raise ToolCallError(msg)
        request = SearchRequest(
            image=photo,
            text=query[:MAX_TEXT],
            filters=self.filters(filters or FilterArgs()),
            position=position,
        )
        result = await _engine(self.site.backend.search(request))
        hits = [self._hit(card) for card in result.cards[: max(1, min(top_k, MAX_TOP_K))]]
        notes: list[str] = []
        if result.words_found is False:
            notes.append(
                "No record holds these words: results come from their meaning alone. "
                "Names and place names work best."
            )
        if result.caption:
            notes.append(f"Caption read on the archive print and searched: «{result.caption}».")
        if result.nearby is not None:
            notes.append(f"{result.nearby} known places within {result.near_km} km rank higher.")
        logger.info(
            "mcp %s search image=%s text=%r filters=%s top=%s",
            caller,
            hashlib.sha256(photo).hexdigest()[:12] if photo else "-",
            request.text[:80],
            request.filters,
            [h.id for h in hits[:5]],
        )
        return SearchResult(results=hits, mode=result.mode, notes=notes)

    # --- fetch -----------------------------------------------------------------------------

    async def fetch(self, caller: str, record_id: str) -> Document:
        self._allow(caller, self.call_limit)
        entity_id = await asyncio.to_thread(self._entity_of, record_id.strip())
        page = None if entity_id is None else entity_page(self.catalog, entity_id)
        if entity_id is None or page is None:
            msg = f"no building or object with id {record_id!r}: use an id from `search`"
            raise ToolCallError(msg)
        description = page.description.strip()
        lines = [f"{f.label} : {f.value}" for f in page.facts]
        text = "\n\n".join(p for p in (description, "\n".join(lines)) if p)
        logger.info("mcp %s fetch %s", caller, entity_id)
        return Document(
            id=entity_id,
            title=page.title,
            text=text,
            url=page.pop_url,
            metadata=RecordMetadata(
                collection=page.collection_label,
                place=page.place,
                facts={f.label: f.value for f in page.facts},
                records=[
                    LinkedRecord(ref=r.ref, title=r.title, url=r.pop_url) for r in page.records
                ],
                images=self._shown_images(entity_id, None)[:FETCH_IMAGES],
                photo_count=page.photo_count,
            ),
        )

    def _entity_of(self, record_id: str) -> str | None:
        """An entity id as given, or the entity of a POP record reference."""
        if record_id in self.catalog.entity_row:
            return record_id
        return next((r.entity_id for r in self.catalog.records if r.ref == record_id), None)

    # --- delegate --------------------------------------------------------------------------

    async def delegate(
        self,
        caller: str,
        *,
        request: str = "",
        conversation_id: str | None = None,
        reply: str = "",
        image_url: str | None = None,
        image_base64: str | None = None,
        region_image_url: str | None = None,
        filters: FilterArgs | None = None,
        position: Position | None = None,
        allow_questions: bool = True,
        wait_seconds: int = AGENT_WAIT_S,
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentAnswer:
        agent = self.agent
        if agent is None:
            msg = "MIROIR's agent is not enabled on this server; use `search`"
            raise ToolCallError(msg)
        session = f"mcp:{caller}"
        if conversation_id:
            c = agent.get(session, conversation_id)
            if c is None:
                msg = "unknown or expired conversation_id: start again without it"
                raise ToolCallError(msg)
            region = await self.photo(region_image_url, None)
            if reply.strip() or region is not None:
                self._allow(caller, self.agent_limit)
                kind = Visit.MESSAGE
                if region is not None:
                    kind = Visit.REGION
                elif c.pending is not None:
                    kind = Visit.ANSWER if c.pending.kind is Wait.ANSWER else Visit.MESSAGE
                given = VisitorInput(kind=kind, text=reply, region=region, position=position)
                self._start(agent, c, given)
        else:
            photo = await self.photo(image_url, image_base64)
            if photo is None and not request.strip():
                msg = "give a request (what to find), an image_url, or both"
                raise ToolCallError(msg)
            self._allow(caller, self.agent_limit)
            c = agent.open(
                session,
                photo=photo,
                words="" if photo is not None else request,
                position=position,
                oneshot=not allow_questions,
                filters=self.filters(filters or FilterArgs()),
            )
            self._start(agent, c, VisitorInput(kind=Visit.GOAL, text=request, position=position))
            logger.info("mcp %s delegate opened %s", caller, c.id[:8])
        start = self._turn_start.get(c.id, 0)
        wait = max(1, min(wait_seconds, MAX_AGENT_WAIT_S))
        await _wait_for_turn(c, start, wait, progress)
        return self._answer(c, start)

    def _start(self, agent: ChatService, c: Conversation, given: VisitorInput) -> None:
        try:
            self._turn_start[c.id] = agent.send(c, given)
        except BusyError as err:
            msg = "the agent is still working: call again with only conversation_id"
            raise ToolCallError(msg) from err
        if len(self._turn_start) > 1000:  # forget old conversations now and then
            self._turn_start = {c.id: self._turn_start[c.id]}

    def _answer(self, c: Conversation, start: int) -> AgentAnswer:
        events = c.events[start:]
        said: list[str] = []
        steps: list[str] = []
        picks: list[RecordHit] = []
        found: str | None = None
        question: str | None = None
        choices: list[str] = []
        status: AgentStatus = "running"
        for event in events:
            kind = event.get("type")
            text = str(event.get("text") or "")
            if kind == "step" and event.get("state") == "done":
                steps.append(_step_line(event))
            elif kind == "say" and text:
                said.append(text)
            elif kind == "results":
                found = None if event.get("found") is None else str(event.get("found"))
                picks = self._picks(c, event.get("cards"))
            elif kind == "question":
                question = text
                raw = event.get("choices")
                choices = [str(x) for x in raw] if isinstance(raw, list) else []
            elif kind == "region":
                question = text or "Which part of the photo matters?"
            elif kind == "limit":
                said.append(text)
                status = "stopped"
            elif kind == "error":
                said.append(text)
                status = "error"
            elif kind == "end" and status == "running":
                status = _ENDS.get(str(event.get("waiting")), "done")
        return AgentAnswer(
            conversation_id=c.id,
            status=status,
            summary="\n".join(said),
            found=found,
            results=picks,
            question=question,
            choices=choices,
            steps=steps[-STEPS_SHOWN:],
        )

    def _picks(self, c: Conversation, cards: object) -> list[RecordHit]:
        out: list[RecordHit] = []
        listed = cast("list[object]", cards) if isinstance(cards, list) else []
        for item in listed:
            if not isinstance(item, dict):
                continue
            raw = cast("dict[str, object]", item)
            entity_id = str(raw.get("entity_id"))
            known = c.known.get(entity_id)
            if known is None:
                continue
            level = Level(known.level) if known.level else Level.LIKENESS
            hit = self._record_hit(
                entity_id, known.pop_url, known.image_url, level, known.photo_count
            )
            if hit is not None:
                reason = str(raw.get("reason") or "")
                out.append(hit.model_copy(update={"reason": reason}))
        return out

    # --- shared ----------------------------------------------------------------------------

    def _allow(self, caller: str, limit: auth.RateLimit) -> None:
        if not limit.allow(caller):
            msg = f"too many calls: at most {limit.limit} a minute. Wait a minute."
            raise ToolCallError(msg)

    async def photo(self, url: str | None, data: str | None) -> bytes | None:
        """The caller's image, from a link or inline, checked; None when neither is given."""
        if url and data:
            msg = "give image_url or image_base64, not both"
            raise ToolCallError(msg)
        try:
            if url:
                return checked_photo(await self.photos.fetch(url))
            if data:
                return checked_photo(decode_base64(data))
        except PhotoError as err:
            raise ToolCallError(str(err)) from err
        return None

    def filters(self, given: FilterArgs) -> Filters:
        """The site's filters, each value matched to a menu value (case and accents aside)."""
        options = self.site.facet_options
        unknown = [c for c in given.collections if c not in COLLECTION_LABELS]
        if unknown:
            msg = f"unknown collection {unknown[0]!r}; valid: {', '.join(COLLECTION_LABELS)}"
            raise ToolCallError(msg)
        return Filters(
            collections=tuple(dict.fromkeys(given.collections)),
            region=_menu("region", facets.region(given.region), options.regions),
            departement=_menu("departement", given.departement, options.departements.get("", ())),
            domain=_menu("domain", given.domain, options.domains),
            period=_menu("period", given.period, tuple(FacetValue(p, 0) for p in facets.PERIODS)),
        )

    def _hit(self, card: Card) -> RecordHit:
        hit = self._record_hit(
            card.entity_id, card.pop_url, card.image_url, card.level, card.photo_count
        )
        assert hit is not None  # the engine only makes cards of entities with records
        return hit

    def _record_hit(
        self, entity_id: str, pop_url: str, image_url: str | None, level: Level, photos: int
    ) -> RecordHit | None:
        row = self.catalog.entity_row.get(entity_id)
        records = [] if row is None else self.catalog.entity_records(row)
        if not records:
            return None
        main = next((r for r in records if r.pop_url == pop_url), records[0])
        shown = self._shown_images(entity_id, image_url)
        return RecordHit(
            id=entity_id,
            title=main.display_title,
            url=main.pop_url,
            collection=COLLECTION_LABELS.get(main.collection, main.collection),
            place=main.place,
            period=list(main.period),
            authors=_authors(records),
            level=level.value,
            level_label=LEVEL_LABELS[level],
            thumbnail_url=shown[0] if shown else None,
            photo_count=photos,
        )

    def _image_rows(self, entity_id: str) -> list[int]:
        row = self.catalog.entity_row.get(entity_id)
        return [] if row is None else [int(r) for r in self.catalog.entity_image_rows(row)]

    def _shown_images(self, entity_id: str, preferred: str | None) -> list[str]:
        """Card-size URLs of the entity's publishable POP images: the one the search matched
        first when it is publishable, then the cover, then the others."""
        images = self.catalog.images
        rows = [
            r
            for r in self._image_rows(entity_id)
            if images[r].source is Source.POP and self.publishable.allows(images[r].image_id)
        ]
        urls = [display_url(images[r], CARD_SIDE) for r in rows]
        order = sorted(
            range(len(rows)), key=lambda i: (urls[i] != preferred, not images[rows[i]].is_cover)
        )
        return [urls[i] for i in order]


def _authors(records: Sequence[CatalogRecord]) -> list[str]:
    return sorted({a for r in records for a in r.authors})[:6]


def _menu(name: str, value: str | None, options: Sequence[FacetValue]) -> str | None:
    if value is None or not value.strip():
        return None
    wanted = facets.fold(value)
    for option in options:
        if facets.fold(option.value) == wanted:
            return option.value
    close = [o.value for o in options if wanted[:4] and facets.fold(o.value).startswith(wanted[:4])]
    listed = close or [o.value for o in options]
    msg = f"unknown {name} {value!r}; valid values include: {', '.join(listed[:40])}"
    raise ToolCallError(msg)


def _step_line(event: Mapping[str, object]) -> str:
    line = str(event.get("label") or "")
    summary = str(event.get("summary") or "")
    if summary:
        line += f" : {summary}"
    why = str(event.get("why") or "")
    if why:
        line += f" (pourquoi : {why})"
    return line


async def _wait_for_turn(
    c: Conversation,
    start: int,
    seconds: float,
    progress: Callable[[str], Awaitable[None]] | None,
) -> None:
    """Until the turn that started at event `start` ends, or `seconds` pass. Each step the
    agent starts is reported to `progress`."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    n = start
    while True:
        waiter = c.changed
        ended = False
        while n < len(c.events):
            event = c.events[n]
            n += 1
            ended = ended or event.get("type") == "end"
            if progress is not None and event.get("type") == "step":
                await progress(str(event.get("label") or ""))
        if ended or not c.running:
            return
        left = deadline - loop.time()
        if left <= 0:
            return
        try:
            await asyncio.wait_for(waiter.wait(), left)
        except TimeoutError:
            return


async def _engine[T](work: Awaitable[T]) -> T:
    """The engine's answer, or a failure the caller can act on (as `app._wait_for`)."""
    call = asyncio.ensure_future(work)
    try:
        return await asyncio.wait_for(asyncio.shield(call), SEARCH_WAIT_S)
    except TimeoutError as err:
        msg = "MIROIR's search engine is starting: try again in a minute"
        raise ToolCallError(msg) from err
    except ValueError as err:
        logger.info("mcp search refused: %s", err)
        msg = "MIROIR could not read this image"
        raise ToolCallError(msg) from err
