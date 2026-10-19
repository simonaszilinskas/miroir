"""The tools of the conversation page (step H5): each one's name, description, JSON schema and
handler, defined once here, so they could later be served as an MCP server unchanged.

A handler takes the conversation (`ToolContext`, kept by `miroir.web.chat`) and the
arguments the model wrote, and returns a `ToolOutcome`: the text the model reads back, the
step the page shows, thumbnails for the model to look at, and whether to stop and wait for
the visitor (a question, a region to frame) or to show results.

What H4's 10-photo test taught is built in: filters favour by default and cut only when the
model says "hard"; an unknown filter key or value is an error naming the valid ones; a search
that repeats an earlier one, or nearly, is not run and the model is told so.

What the first hosted test taught (a gold Roman dish, not in the index: the model never looked
at the close first results, then drifted on long guessed text queries) is built in too: every
search and look carries a `why` the visitor reads; a text query holds at most 5 words;
`more_like` searches from a record already seen instead of from guessed words; `show_results`
says whether the very thing was found (`found`), and says so plainly when it was not.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from miroir.eval.bm25 import fold, tokens
from miroir.jobs.openrouter import JSON
from miroir.records.bases import Base
from miroir.web.nearby import Position
from miroir.web.toolsearch import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    Hit,
    Near,
    SearchKind,
    ToolFilters,
    ToolHits,
    ToolQuery,
)

logger = logging.getLogger(__name__)

#: Radius of a `near` filter when the model gives none, and the largest it may ask for.
NEAR_KM = 2.0
MAX_NEAR_KM = 50.0
#: Most entities one view shows the model, and most results it may show the visitor.
MAX_VIEW = 8
MAX_RESULTS = 10
#: Suggested answers to a question.
MIN_CHOICES = 2
MAX_CHOICES = 4
#: Two text searches with the same filters whose word sets overlap this much are one search.
SAME_WORDS = 0.75
#: Thumbnails a step shows the visitor.
STEP_THUMBS = 8
#: Characters of a title in a result line.
TITLE_CHARS = 90
#: Most words in a text query (the hosted test's drifting queries ran to 10 and more).
MAX_QUERY_WORDS = 5
#: Characters kept of a `why`, of a summary.
WHY_CHARS = 200
SUMMARY_CHARS = 700
#: What a "not found" summary starts with when the model did not say it.
NOT_FOUND = "Je ne trouve pas cet objet précis dans les collections."


class ToolName(StrEnum):
    IMAGE_SEARCH = "image_search"
    TEXT_SEARCH = "text_search"
    VIEW_RESULTS = "view_results"
    MORE_LIKE = "more_like"
    ASK_VISITOR = "ask_visitor"
    ASK_REGION = "ask_region"
    SHOW_RESULTS = "show_results"


#: The searches: each one counts towards the turn's searches.
SEARCHES = frozenset({ToolName.IMAGE_SEARCH, ToolName.TEXT_SEARCH, ToolName.MORE_LIKE})


class Found(StrEnum):
    """What `show_results` says it found."""

    EXACT = "exact"
    """The very object or building: checked by eye, or rated "Très proche" by the site."""
    CLOSE = "close"
    """No sure match: the closest the collections hold."""
    NONE = "none"


class Wait(StrEnum):
    """What the page waits for from the visitor."""

    ANSWER = "answer"
    REGION = "region"


# --- what a handler returns ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepThumb:
    entity_id: str
    title: str
    image_url: str | None


@dataclass(frozen=True, slots=True)
class Step:
    """One tool call as the page shows it: a line, a count, some thumbnails."""

    label: str
    summary: str
    thumbs: tuple[StepThumb, ...] = ()
    error: bool = False


@dataclass(frozen=True, slots=True)
class Thumbnail:
    """An image for the model to look at, already checked as publishable and fetched."""

    label: str
    image_id: str


@dataclass(frozen=True, slots=True)
class Question:
    text: str
    choices: tuple[str, ...]
    offer_position: bool = False


@dataclass(frozen=True, slots=True)
class Pick:
    entity_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    text: str
    """What the model reads back."""
    step: Step | None = None
    thumbnails: tuple[Thumbnail, ...] = ()
    wait: Wait | None = None
    question: Question | None = None
    region_reason: str = ""
    picks: tuple[Pick, ...] | None = None
    """show_results: the entities to show, best first, only ones the model has seen."""
    found: Found | None = None
    summary: str = ""
    """show_results: what the visitor reads above the cards."""
    hit_ids: tuple[str, ...] = ()
    """The entities a search or a look listed, in order."""
    error: bool = False
    counted: bool = True
    """It used one of the turn's tool calls (searches and views do; questions do not)."""


class ToolContext(Protocol):
    """The conversation, as the tools see it."""

    @property
    def photo(self) -> bytes | None: ...
    @property
    def region(self) -> bytes | None: ...
    @property
    def position(self) -> Position | None: ...
    async def search(self, query: ToolQuery) -> ToolHits: ...
    def remember(self, hits: Sequence[Hit]) -> None:
        """Entities the model has now seen (show_results takes only those)."""
        ...

    def seen(self, entity_id: str) -> bool: ...
    async def thumbnail(self, hit: Hit) -> str | None:
        """The image id of the hit's first image that may go out and could be fetched."""
        ...

    def repeats(self, signature: Signature) -> int | None:
        """The step number of an earlier call this one repeats, else None (and noted)."""
        ...

    def viewed_image(self, entity_id: str) -> str | None:
        """The image of the entity the model has looked at with view_results, if any."""
        ...

    def checked(self, entity_id: str) -> bool:
        """The model has looked at its thumbnail, or the site rated it "Très proche"."""
        ...

    def title_of(self, entity_id: str) -> str: ...


type Handler = Callable[[ToolContext, Mapping[str, object]], Awaitable[ToolOutcome]]


@dataclass(frozen=True, slots=True)
class Tool:
    name: ToolName
    description: str
    parameters: dict[str, JSON]
    handler: Handler
    needs_photo: bool = False
    needs_why: bool = False
    """It takes a `why`, and is refused without one (the searches and the looks)."""

    def schema(self) -> JSON:
        """The tool as an OpenAI-style function (OpenRouter, Mistral)."""
        return {
            "type": "function",
            "function": {
                "name": self.name.value,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ArgumentError(ValueError):
    """The model's arguments cannot be used; the message tells it why."""


# --- filters ---------------------------------------------------------------------------------

FILTER_KEYS = ("base", "departement", "commune", "domain", "period", "near", "mode")
_BASES = tuple(b.value for b in Base)

_FILTERS_SCHEMA: dict[str, JSON] = {
    "type": "object",
    "description": (
        "Favour records matching every field given (mode boost, the default: +0.25 on a 0-1 "
        "score), or keep only them (mode hard: only when you are sure, a hard filter on a "
        "wrong guess loses the answer)."
    ),
    "properties": {
        "base": {"type": "array", "items": {"type": "string", "enum": list(_BASES)}},
        "departement": {"type": "array", "items": {"type": "string"}},
        "commune": {"type": "array", "items": {"type": "string"}},
        "domain": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Words of the record's domain or kind: église, peinture, vase...",
        },
        "period": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A century or period: '12e siècle', 'Moyen Âge', 'Antiquité'.",
        },
        "near": {
            "description": (
                "Favour (or with hard, keep) records placed within km of a point: "
                '{"lat": ..., "lon": ..., "km": ...}, or "visitor" for the position the '
                f"visitor shared (default {NEAR_KM:g} km)."
            ),
            "anyOf": [
                {"type": "string", "enum": ["visitor"]},
                {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                        "km": {"type": "number"},
                    },
                    "required": ["lat", "lon"],
                },
            ],
        },
        "mode": {"type": "string", "enum": ["boost", "hard"]},
    },
}


def parse_filters(raw: object, position: Position | None) -> ToolFilters:
    """Filters from the model's JSON. Strict: a key or value we do not know is an error that
    names the valid ones (H4 dropped them silently, and the model never knew)."""
    if raw is None:
        return ToolFilters()
    if not isinstance(raw, dict):
        msg = f"filters must be an object with keys among: {', '.join(FILTER_KEYS)}"
        raise ArgumentError(msg)
    given: dict[str, object] = {str(k): v for k, v in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    unknown = [k for k in given if k not in FILTER_KEYS]
    if unknown:
        msg = (
            f"unknown filter key(s): {', '.join(unknown)}. Valid keys: {', '.join(FILTER_KEYS)}. "
            "Nothing was searched; call again with valid keys."
        )
        raise ArgumentError(msg)
    bases: list[Base] = []
    for value in _strings(given.get("base"), "base"):
        key = re.sub(r"[^a-z-]", "", fold(value).replace("_", "-"))
        match = next((b for b in Base if key in (b.value, b.value.replace("-", ""))), None)
        if match is None:
            msg = f"unknown base {value!r}. Valid bases: {', '.join(_BASES)}."
            raise ArgumentError(msg)
        bases.append(match)
    mode = str(given.get("mode", "boost")).strip().lower()
    if mode not in ("boost", "hard"):
        msg = f"unknown mode {mode!r}. Valid modes: boost (the default), hard."
        raise ArgumentError(msg)
    return ToolFilters(
        bases=tuple(dict.fromkeys(bases)),
        departements=_strings(given.get("departement"), "departement"),
        communes=_strings(given.get("commune"), "commune"),
        domains=_strings(given.get("domain"), "domain"),
        periods=_strings(given.get("period"), "period"),
        near=_near(given.get("near"), position),
        hard=mode == "hard",
    )


def _near(value: object, position: Position | None) -> Near | None:
    if value is None:
        return None
    if value == "visitor" or (isinstance(value, dict) and value.get("visitor")):  # pyright: ignore[reportUnknownMemberType]
        if position is None:
            msg = 'near "visitor": the visitor has shared no position. Ask them where they are.'
            raise ArgumentError(msg)
        km = _number(value.get("km") if isinstance(value, dict) else None, NEAR_KM)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        return Near(position.latitude, position.longitude, _radius(km))
    if not isinstance(value, dict):
        msg = 'near must be "visitor" or {"lat": ..., "lon": ..., "km": ...}'
        raise ArgumentError(msg)
    given: dict[str, object] = {str(k): v for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    lat = _number(given.get("lat", given.get("latitude")), float("nan"))
    lon = _number(given.get("lon", given.get("longitude")), float("nan"))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        msg = "near needs lat (-90 to 90) and lon (-180 to 180) as numbers"
        raise ArgumentError(msg)
    return Near(lat, lon, _radius(_number(given.get("km"), NEAR_KM)))


def _radius(km: float) -> float:
    return min(max(km, 0.1), MAX_NEAR_KM)


def describe_filters(filters: ToolFilters) -> str:
    """The filters in the visitor's words, for a step line ("" when none)."""
    if filters.empty:
        return ""
    names = {
        Base.MERIMEE: "bâtiments",
        Base.PALISSY: "objets d’église",
        Base.JOCONDE: "musées",
        Base.ROSE_VALLAND: "Rose Valland",
        Base.MEMOIRE: "photos Mémoire",
    }
    parts = [names[b] for b in filters.bases]
    parts += [*filters.communes, *filters.departements, *filters.domains, *filters.periods]
    if filters.near is not None:
        parts.append(f"à moins de {filters.near.km:g} km".replace(".", ","))
    word = "seulement" if filters.hard else "en priorité"
    return f"{word} : {', '.join(parts)}"


def _filters_key(filters: ToolFilters) -> str:
    near = filters.near
    return json.dumps(
        [
            sorted(b.value for b in filters.bases),
            sorted(tokens(" ".join(filters.departements))),
            sorted(tokens(" ".join(filters.communes))),
            sorted(tokens(" ".join(filters.domains))),
            sorted(tokens(" ".join(filters.periods))),
            None if near is None else [round(near.latitude, 3), round(near.longitude, 3), near.km],
            filters.hard,
        ]
    )


# --- repeats ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signature:
    """What makes two calls the same search."""

    tool: ToolName
    key: str
    """Everything but the words, as JSON."""
    words: frozenset[str] = frozenset()

    def same(self, other: Signature) -> bool:
        if self.tool is not other.tool or self.key != other.key:
            return False
        if not self.words and not other.words:
            return True
        shared = len(self.words & other.words)
        return shared / max(len(self.words | other.words), 1) >= SAME_WORDS


def _repeated(step: int, what: str) -> ToolOutcome:
    return ToolOutcome(
        f"not run: this repeats step {step} ({what}). Its results are above. Change the "
        "approach: another image model, other filters, a question to the visitor, or "
        "show_results with what you have.",
        Step("Recherche déjà faite", "pas relancée", error=True),
        error=True,
    )


# --- handlers --------------------------------------------------------------------------------


async def image_search(ctx: ToolContext, args: Mapping[str, object]) -> ToolOutcome:
    source = str(args.get("source", "photo")).strip().lower()
    if source not in ("photo", "region"):
        msg = f"unknown source {source!r}: photo or region"
        raise ArgumentError(msg)
    image = ctx.region if source == "region" else ctx.photo
    if image is None:
        what = "framed region: call ask_region first" if source == "region" else "photo"
        msg = f"there is no {what}"
        raise ArgumentError(msg)
    model = str(args.get("model", "auto")).strip().lower()
    weights = {"auto": None, "a7": 1.0, "a3": 0.0}
    if model == "mix":
        weight: float | None = min(max(_number(args.get("a7_weight"), 0.5), 0.0), 1.0)
    elif model in weights:
        weight = weights[model]
    else:
        msg = f"unknown model {model!r}: auto, A7, A3 or mix"
        raise ArgumentError(msg)
    filters = parse_filters(args.get("filters"), ctx.position)
    signature = Signature(
        ToolName.IMAGE_SEARCH,
        json.dumps([source, weight, _filters_key(filters), len(ctx.region or b"")]),
    )
    if (step := ctx.repeats(signature)) is not None:
        return _repeated(step, "the same image search")
    found = await ctx.search(
        ToolQuery(
            SearchKind.IMAGE,
            image=image,
            a7_weight=weight,
            filters=filters,
            top_k=_top_k(args),
        )
    )
    names = {None: "réglage automatique", 1.0: "EmbeddingGemma 2", 0.0: "MegaLoc"}
    used = names.get(weight, f"mélange, {round(100 * (weight or 0))} % EmbeddingGemma 2")
    label = f"Recherche par image ({used})"
    if source == "region":
        label += ", sur la partie encadrée"
    head = f"image_search {source}, model {model}"
    if found.a7_weight is not None:
        head += f" (A7 weight {found.a7_weight:g}, photo routed as {found.bucket})"
    if found.mount:
        head += "; an archive print on its card: the print alone was searched"
    return _listing(ctx, head, label, filters, found)


async def text_search(ctx: ToolContext, args: Mapping[str, object]) -> ToolOutcome:
    query = str(args.get("query") or "").strip()
    if not query:
        msg = "text_search needs a query, in French"
        raise ArgumentError(msg)
    if len(query.split()) > MAX_QUERY_WORDS:
        msg = (
            f"a text query holds 2 to {MAX_QUERY_WORDS} words, taken from the visitor's words, "
            "a caption you read in the photo, or titles of results you have seen. Nothing was "
            "searched."
        )
        raise ArgumentError(msg)
    filters = parse_filters(args.get("filters"), ctx.position)
    signature = Signature(ToolName.TEXT_SEARCH, _filters_key(filters), frozenset(tokens(query)))
    if (step := ctx.repeats(signature)) is not None:
        return _repeated(step, "a text search with nearly the same words and filters")
    found = await ctx.search(
        ToolQuery(SearchKind.TEXT, text=query, filters=filters, top_k=_top_k(args))
    )
    label = f"Recherche par mots « {query[:60]} »"
    if found.words_found is False:
        return ToolOutcome(
            f"text_search {query!r}: no record holds these words. Try other French words "
            "(as records write them) or another approach.",
            Step(label, "aucune notice ne contient ces mots"),
        )
    return _listing(ctx, f"text_search {query!r}", label, filters, found)


def _listing(
    ctx: ToolContext, head: str, label: str, filters: ToolFilters, found: ToolHits
) -> ToolOutcome:
    ctx.remember(found.hits)
    described = describe_filters(filters)
    if described:
        label += f", {described}"
    if found.matching is not None:
        head += f"; filters ({'hard' if filters.hard else 'boost'}): {found.matching} match"
    if found.near_ignored:
        head += "; near ignored: the index holds no places"
    lines = [f"{i}. {hit_line(h)}" for i, h in enumerate(found.hits, start=1)]
    body = "\n".join(lines) if lines else "nothing found"
    n = len(found.hits)
    summary = f"{n} résultat{'s' if n > 1 else ''}" if n else "aucun résultat"
    thumbs = tuple(StepThumb(h.entity_id, h.title, h.image_url) for h in found.hits[:STEP_THUMBS])
    return ToolOutcome(
        f"{head}\nid | score | base | title | place | authors | date\n{body}",
        Step(label, summary, thumbs),
        hit_ids=tuple(h.entity_id for h in found.hits),
    )


def hit_line(hit: Hit, *, score: bool = True) -> str:
    parts = [hit.entity_id]
    if score:
        parts.append(f"{hit.score:.2f}" + (" *" if hit.matches else ""))
    parts += [hit.base, hit.title[:TITLE_CHARS], hit.place, hit.authors, hit.period]
    return " | ".join(parts)


async def view_results(ctx: ToolContext, args: Mapping[str, object]) -> ToolOutcome:
    asked = list(dict.fromkeys(_strings(args.get("entity_ids"), "entity_ids")))
    if not asked:
        msg = "view_results needs entity_ids from earlier results"
        raise ArgumentError(msg)
    unseen = [e for e in asked if not ctx.seen(e)]
    wanted = [e for e in asked if ctx.seen(e)][:MAX_VIEW]
    if not wanted:
        msg = f"none of these ids came from a search: {', '.join(unseen[:5])}"
        raise ArgumentError(msg)
    signature = Signature(ToolName.VIEW_RESULTS, json.dumps(sorted(wanted)))
    if (step := ctx.repeats(signature)) is not None:
        return _repeated(step, "the same thumbnails")
    found = await ctx.search(ToolQuery(SearchKind.VIEW, image=ctx.photo, entity_ids=tuple(wanted)))
    lines: list[str] = []
    thumbnails: list[Thumbnail] = []
    fetched = await asyncio.gather(*(ctx.thumbnail(hit) for hit in found.hits))
    for i, (hit, image_id) in enumerate(zip(found.hits, fetched, strict=True), start=1):
        if image_id is None:
            lines.append(f"[{i}] {hit_line(hit, score=False)} (no image may be shown)")
            continue
        lines.append(f"[{i}] {hit_line(hit, score=False)}")
        thumbnails.append(Thumbnail(f"[{i}] {hit.entity_id}", image_id))
    if unseen:
        lines.append(f"left out, never seen in a result: {', '.join(unseen[:5])}")
    if len(asked) - len(unseen) > MAX_VIEW:
        lines.append(f"only the first {MAX_VIEW} are shown")
    n = len(thumbnails)
    thumbs = tuple(StepThumb(h.entity_id, h.title, h.image_url) for h in found.hits)
    return ToolOutcome(
        "view_results: thumbnails follow in the next message, closest image first\n"
        + "\n".join(lines),
        Step("Regarde des images de près", f"{n} image{'s' if n > 1 else ''}", thumbs),
        thumbnails=tuple(thumbnails),
        hit_ids=tuple(h.entity_id for h in found.hits),
    )


async def more_like(ctx: ToolContext, args: Mapping[str, object]) -> ToolOutcome:
    entity_id = str(args.get("entity_id") or args.get("id") or "").strip()
    if not entity_id:
        msg = "more_like needs entity_id: a record you have seen in a result"
        raise ArgumentError(msg)
    if not ctx.seen(entity_id):
        msg = f"{entity_id} never came from a search: start from an id you have seen"
        raise ArgumentError(msg)
    model = str(args.get("model", "mix")).strip().lower()
    if model == "mix":
        weight = min(max(_number(args.get("a7_weight"), 0.5), 0.0), 1.0)
    elif model in ("a7", "a3"):
        weight = 1.0 if model == "a7" else 0.0
    else:
        msg = f"unknown model {model!r}: mix (the default), A7 or A3"
        raise ArgumentError(msg)
    filters = parse_filters(args.get("filters"), ctx.position)
    signature = Signature(
        ToolName.MORE_LIKE, json.dumps([entity_id, weight, _filters_key(filters)])
    )
    if (step := ctx.repeats(signature)) is not None:
        return _repeated(step, "the same search by look")
    image_id = ctx.viewed_image(entity_id)
    found = await ctx.search(
        ToolQuery(
            SearchKind.LIKE,
            entity_ids=(entity_id,),
            image_id=image_id,
            a7_weight=weight,
            filters=filters,
            top_k=_top_k(args),
        )
    )
    title = ctx.title_of(entity_id) or entity_id
    label = f"Ce qui ressemble à « {title[:50]} »"
    start = "the image you looked at" if image_id else "its cover image"
    head = f"more_like {entity_id} (from {start}; A7 weight {weight:g}; itself left out)"
    return _listing(ctx, head, label, filters, found)


async def ask_visitor(ctx: ToolContext, args: Mapping[str, object]) -> ToolOutcome:
    del ctx
    question = str(args.get("question") or "").strip()
    if not question:
        msg = "ask_visitor needs a question, in French"
        raise ArgumentError(msg)
    # The page always offers "Je ne sais pas" itself.
    given = _strings(args.get("choices"), "choices")
    choices = tuple(dict.fromkeys(c[:80] for c in given if "ne sais pas" not in fold(c)))
    if not MIN_CHOICES <= len(choices) <= MAX_CHOICES:
        msg = f"ask_visitor needs {MIN_CHOICES} to {MAX_CHOICES} short suggested answers"
        raise ArgumentError(msg)
    offer = args.get("offer_position") is True
    return ToolOutcome(
        "",
        wait=Wait.ANSWER,
        question=Question(question[:300], choices, offer),
        counted=False,
    )


async def ask_region(ctx: ToolContext, args: Mapping[str, object]) -> ToolOutcome:
    if ctx.photo is None:
        msg = "there is no photo to frame"
        raise ArgumentError(msg)
    reason = str(args.get("reason") or "").strip()[:300]
    return ToolOutcome("", wait=Wait.REGION, region_reason=reason, counted=False)


async def show_results(ctx: ToolContext, args: Mapping[str, object]) -> ToolOutcome:  # noqa: PLR0912 (one check per rule)
    try:
        found = Found(str(args.get("found") or "").strip().lower())
    except ValueError:
        msg = (
            "show_results needs found: exact (one pick is the very thing, and you have seen "
            "its thumbnail), close (no sure match: these are the closest) or none"
        )
        raise ArgumentError(msg) from None
    summary = " ".join(str(args.get("summary") or "").split())[:SUMMARY_CHARS]
    if found is not Found.EXACT and not summary:
        msg = (
            f"with found {found.value}, write summary: one or two French sentences the visitor "
            f"reads above the cards, saying plainly what was not found and why these are the "
            f"closest (« {NOT_FOUND} Les plus proches : …, parce que … »)"
        )
        raise ArgumentError(msg)
    if found is Found.NONE and "trouve pas" not in fold(summary):
        summary = f"{NOT_FOUND} {summary}"
    raw: object = args.get("results") or []
    if not isinstance(raw, list) or (not raw and found is not Found.NONE):
        msg = 'show_results needs results: [{"id": ..., "reason": ...}, ...], best first'
        raise ArgumentError(msg)
    picks: list[Pick] = []
    dropped: list[str] = []
    for item in raw:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(item, dict):
            entity_id = str(item.get("id") or item.get("entity_id") or "").strip()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            reason = str(item.get("reason") or "").strip()[:200]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        else:
            entity_id, reason = str(item).strip(), ""  # pyright: ignore[reportUnknownArgumentType]
        if not entity_id or any(p.entity_id == entity_id for p in picks):
            continue
        if ctx.seen(entity_id):
            picks.append(Pick(entity_id, reason))
        else:
            dropped.append(entity_id)
    if not picks and found is not Found.NONE:
        msg = f"none of these ids came from a search ({', '.join(dropped[:5])}); use ids you saw"
        raise ArgumentError(msg)
    if found is Found.EXACT and not ctx.checked(picks[0].entity_id):
        msg = (
            f"found exact needs a first pick you have checked by eye: call view_results on "
            f"{picks[0].entity_id} first, or use found close"
        )
        raise ArgumentError(msg)
    kept = picks[:MAX_RESULTS]
    text = f"shown to the visitor: {len(kept)} result(s)"
    if dropped:
        text += f"; left out, never seen in a result: {', '.join(dropped[:5])}"
        logger.info("show_results dropped %d id(s) never seen", len(dropped))
    return ToolOutcome(text, picks=tuple(kept), found=found, summary=summary, counted=False)


# --- the registry ----------------------------------------------------------------------------

_TOP_K: JSON = {"type": "integer", "description": f"1 to {MAX_TOP_K}, default {DEFAULT_TOP_K}"}
_WHY: JSON = {
    "type": "string",
    "description": "Required. One short sentence in French, shown to the visitor: what the last "
    "result showed, and what this call tests (« Les pièces d'or ressemblent aux médaillons du "
    "bord ; je les regarde de près. »).",
}
_BEST: JSON = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Up to 3 ids you have seen that are the closest so far, best first ([] when "
    "none fits). Kept across steps; leave it out to keep it as it is.",
}

TOOLS: tuple[Tool, ...] = (
    Tool(
        ToolName.IMAGE_SEARCH,
        "Search by the look of the visitor's photo (or of the part they framed). Lists "
        "entities: id | score | base | title | place | authors | date; * marks a filter match.",
        {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["photo", "region"]},
                "model": {
                    "type": "string",
                    "enum": ["auto", "A7", "A3", "mix"],
                    "description": "auto: the site's choice for this photo. A7 (EmbeddingGemma "
                    "2) for black and white archive prints of buildings; A3 (MegaLoc) for "
                    "colour photos of buildings and for objects; mix takes a7_weight.",
                },
                "a7_weight": {"type": "number", "description": "For mix: 0 to 1."},
                "top_k": _TOP_K,
                "filters": _FILTERS_SCHEMA,
                "why": _WHY,
                "best": _BEST,
            },
            "required": ["why"],
        },
        image_search,
        needs_photo=True,
        needs_why=True,
    ),
    Tool(
        ToolName.TEXT_SEARCH,
        "Search record text (titles, descriptions, communes, départements, periods, authors) "
        f"with BM25. 2 to {MAX_QUERY_WORDS} French words as records write them, taken from the "
        "visitor's words, a caption you read in the photo, or titles of results you have seen; "
        "never a name or place you only guessed.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": _TOP_K,
                "filters": _FILTERS_SCHEMA,
                "why": _WHY,
                "best": _BEST,
            },
            "required": ["query", "why"],
        },
        text_search,
        needs_why=True,
    ),
    Tool(
        ToolName.VIEW_RESULTS,
        f"Look at thumbnails (256 px) of up to {MAX_VIEW} entities from earlier results, each "
        "its image closest to the photo, to compare them with the photo by eye.",
        {
            "type": "object",
            "properties": {
                "entity_ids": {"type": "array", "items": {"type": "string"}},
                "why": _WHY,
                "best": _BEST,
            },
            "required": ["entity_ids", "why"],
        },
        view_results,
        needs_why=True,
    ),
    Tool(
        ToolName.MORE_LIKE,
        "Search for records that look like one you have already seen in a result, by its own "
        "stored image (the one you looked at with view_results, else its cover). Use it to "
        "follow a close result instead of guessing words. Lists entities like image_search; "
        "the record itself is left out.",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "model": {
                    "type": "string",
                    "enum": ["mix", "A7", "A3"],
                    "description": "mix (the default, a7_weight 0.5): both. A7 (EmbeddingGemma "
                    "2): the same kind of thing. A3 (MegaLoc): the same building or view.",
                },
                "a7_weight": {"type": "number", "description": "For mix: 0 to 1."},
                "top_k": _TOP_K,
                "filters": _FILTERS_SCHEMA,
                "why": _WHY,
                "best": _BEST,
            },
            "required": ["entity_id", "why"],
        },
        more_like,
        needs_why=True,
    ),
    Tool(
        ToolName.ASK_VISITOR,
        "Ask the visitor one short question in French, in plain words, with 2 to 4 suggested "
        "answers. They may also type their own answer or say they do not know. Set "
        "offer_position when where they stand would help.",
        {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}},
                "offer_position": {"type": "boolean"},
            },
            "required": ["question", "choices"],
        },
        ask_visitor,
    ),
    Tool(
        ToolName.ASK_REGION,
        "Ask the visitor to frame the part of the photo that matters (one painting on a wall, "
        "one object in a room). The framed part comes back as an image; image_search then "
        "takes source region.",
        {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why, in French."}},
        },
        ask_region,
        needs_photo=True,
    ),
    Tool(
        ToolName.SHOW_RESULTS,
        f"Show the visitor your picks, best first, up to {MAX_RESULTS}, each with one short "
        "reason in French. Only ids you have seen in results. Say honestly what you found. "
        "Ends your turn.",
        {
            "type": "object",
            "properties": {
                "found": {
                    "type": "string",
                    "enum": [f.value for f in Found],
                    "description": "exact: the first pick is the very object or building, and "
                    "you have seen its thumbnail. close: no sure match, these are the closest. "
                    "none: nothing fits.",
                },
                "summary": {
                    "type": "string",
                    "description": "One or two French sentences the visitor reads above the "
                    "cards. Required unless found is exact. When the visitor wants one thing "
                    f"and it is not there: « {NOT_FOUND} Les plus proches : …, parce que … »",
                },
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                        "required": ["id", "reason"],
                    },
                },
            },
            "required": ["found", "results"],
        },
        show_results,
    ),
)

BY_NAME: dict[str, Tool] = {t.name.value: t for t in TOOLS}


def tools_for(*, photo: bool) -> tuple[Tool, ...]:
    """The tools a conversation offers: without a photo, no image search, no framing."""
    return tuple(t for t in TOOLS if photo or not t.needs_photo)


async def run_tool(ctx: ToolContext, name: str, arguments: str) -> ToolOutcome:
    """One call, as the model wrote it. Bad arguments come back as an error it can read."""
    tool = BY_NAME.get(name)
    if tool is None:
        known = ", ".join(BY_NAME)
        return ToolOutcome(f"error: no tool named {name!r}. Tools: {known}", error=True)
    try:
        args: object = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return ToolOutcome(f"error: the arguments are not JSON: {arguments[:200]}", error=True)
    if not isinstance(args, dict):
        return ToolOutcome("error: the arguments must be a JSON object", error=True)
    given: dict[str, object] = {str(k): v for k, v in args.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    try:
        if tool.needs_why and not why_of(given):
            msg = (
                f"{name} needs why: one short French sentence saying what the last result "
                "showed and what this call tests. Nothing was run; call again with it."
            )
            raise ArgumentError(msg)
        return await tool.handler(ctx, given)
    except ArgumentError as err:
        return ToolOutcome(
            f"error: {err}", Step(_LABELS[tool.name], str(err), error=True), error=True
        )


def arguments_of(arguments: str) -> dict[str, object]:
    """The model's arguments as a dict; {} when they are not a JSON object."""
    try:
        args: object = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(args, dict):
        return {}
    return {str(k): v for k, v in args.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def why_of(args: Mapping[str, object]) -> str:
    """The call's `why`, on one line ("" when none)."""
    return " ".join(str(args.get("why") or "").split())[:WHY_CHARS]


def best_of(args: Mapping[str, object]) -> tuple[str, ...] | None:
    """The call's `best`: the ids it names, or None when it left the field out."""
    if "best" not in args:
        return None
    try:
        return tuple(dict.fromkeys(_strings(args.get("best"), "best")))
    except ArgumentError:
        return None


_LABELS = {
    ToolName.IMAGE_SEARCH: "Recherche par image refusée",
    ToolName.TEXT_SEARCH: "Recherche par mots refusée",
    ToolName.VIEW_RESULTS: "Vignettes refusées",
    ToolName.MORE_LIKE: "Recherche par ressemblance refusée",
    ToolName.ASK_VISITOR: "Question mal formée",
    ToolName.ASK_REGION: "Cadrage impossible",
    ToolName.SHOW_RESULTS: "Résultats refusés",
}


def _top_k(args: Mapping[str, object]) -> int:
    return int(min(max(_number(args.get("top_k"), DEFAULT_TOP_K), 1), MAX_TOP_K))


def _strings(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | int | float):
        items: list[object] = [value]
    elif isinstance(value, list):
        items = list(value)  # pyright: ignore[reportUnknownArgumentType]
    else:
        msg = f"{name} must be a list of strings"
        raise ArgumentError(msg)
    return tuple(s for s in (str(v).strip() for v in items if v is not None) if s)


def _number(value: object, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(str(value))
    except ValueError:
        return default
    return number if math.isfinite(number) else default
