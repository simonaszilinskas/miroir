"""Step H4: the search tools a vision model calls, over vectors and text we already have.

No model runs here. Every tool reads saved vectors (EmbeddingGemma 2, A7; MegaLoc, A3) and
BM25 over record text (`miroir.eval.vlm`, as in H1):

- `image_search`: cosine search with the query photo's saved vectors, A7, A3, or a weighted
  mix of their min-max scores (as G1 and G5 do);
- `text_search`: BM25 over the gallery's record text, min-max scaled;
- `view_results`: names the gallery image of each entity that best matches the photo, so the
  caller can send its thumbnail;
- `final_answer`: up to 10 entities, best first.

Filters (POP base, département, commune, domain, period) work as a soft boost or a hard cut;
the model picks. The query's hidden images (its own, and for T1 its whole shoot) never score,
never show, and so never reach the model.

`World` holds the gallery and every query's scores; `Session` runs one query's tool calls and
remembers what the model has seen; `Plan` is the one-shot search of scenario S1.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from miroir.eval.bm25 import fold, tokens
from miroir.eval.fusion import min_max
from miroir.eval.scoring import Floats, SearchQuery
from miroir.eval.vlm import TextGallery
from miroir.geo.model import distance_km
from miroir.records.model import Collection, Record

logger = logging.getLogger(__name__)

#: Added to an entity's 0-1 score when it matches a soft filter.
BOOST = 0.25
#: Most entities one search may list, and the default.
MAX_TOP_K = 30
DEFAULT_TOP_K = 10
#: Most entities one view_results call shows, and most a final answer holds.
MAX_VIEW = 10
MAX_ANSWER = 10
#: Characters of a title shown in a result line.
TITLE_CHARS = 90


class Base(StrEnum):
    """Where an entity comes from in POP, as the model filters on it."""

    MERIMEE = "merimee"
    """A building (Mérimée id PA…); its photos are Mémoire records or Commons photos."""
    PALISSY = "palissy"
    """An object kept in a building (Palissy id PM…), photographed in Mémoire."""
    JOCONDE = "joconde"
    """A museum object."""
    MEMOIRE = "memoire"
    """A Mémoire photo tied to no building or object."""
    ROSE_VALLAND = "rose-valland"
    """A work looted during the war (MNR), or a Jeu de Paume room photo."""


def base_of(collection: Collection, entity_id: str) -> Base:
    if collection is Collection.JOCONDE:
        return Base.JOCONDE
    if collection is Collection.ROSE_VALLAND:
        return Base.ROSE_VALLAND
    if entity_id.startswith("PA"):
        return Base.MERIMEE
    if entity_id.startswith("PM"):
        return Base.PALISSY
    return Base.MEMOIRE


def place_key(text: str) -> str:
    """A commune or département name folded for comparison: "Saint-Étienne" = "saint etienne"."""
    return " ".join(re.findall(r"[a-z0-9]+", fold(text)))


@dataclass(frozen=True, slots=True)
class EntityInfo:
    entity_id: str
    base: Base
    title: str
    communes: tuple[str, ...]
    departements: tuple[str, ...]
    domain_words: frozenset[str]
    """Folded words of its records' domains and denominations."""
    period_words: frozenset[str]
    position: tuple[float, float] | None = None
    """Latitude and longitude (`places.parquet`: Mérimée point, museum, or commune centre)."""

    @property
    def place(self) -> str:
        commune = self.communes[0] if self.communes else ""
        dept = self.departements[0] if self.departements else ""
        if commune and dept:
            return f"{commune} ({dept})"
        return commune or dept


def entity_infos(
    records: Sequence[Record],
    entity_ids: Sequence[str],
    positions: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[EntityInfo, ...]:
    """What the tools say about each entity, from every record that shows it."""
    where = positions or {}
    by_entity: dict[str, list[Record]] = {}
    for record in records:
        by_entity.setdefault(record.entity_id, []).append(record)
    out: list[EntityInfo] = []
    for entity_id in entity_ids:
        found = by_entity.get(entity_id, [])
        if not found:
            msg = f"entity {entity_id} has no record"
            raise ValueError(msg)
        out.append(_info(entity_id, sorted(found, key=lambda r: r.ref), where.get(entity_id)))
    return tuple(out)


def _info(
    entity_id: str, records: Sequence[Record], position: tuple[float, float] | None
) -> EntityInfo:
    # A building's own record title names it; a Mémoire photo title often names a part of it.
    titled = [r.title for r in records if r.title]
    title = titled[0] if titled else ""
    if not title:
        named = [
            *(d for r in records for d in r.denomination),
            *(d for r in records for d in r.domain),
        ]
        title = "; ".join(dict.fromkeys(named))
    return EntityInfo(
        entity_id=entity_id,
        base=base_of(records[0].collection, entity_id),
        title=title[:TITLE_CHARS],
        communes=tuple(dict.fromkeys(r.commune for r in records if r.commune)),
        departements=tuple(dict.fromkeys(r.departement for r in records if r.departement)),
        domain_words=frozenset(
            tokens(" ".join(w for r in records for w in (*r.domain, *r.denomination)))
        ),
        period_words=frozenset(tokens(" ".join(p for r in records for p in r.period))),
        position=position,
    )


# --- filters ---------------------------------------------------------------------------------


class FilterMode(StrEnum):
    BOOST = "boost"
    """Matching entities gain `BOOST` on their 0-1 score; the others stay."""
    HARD = "hard"
    """Only matching entities are kept."""


@dataclass(frozen=True, slots=True)
class Filters:
    """Every field given must match (one of its values); an empty field matches all."""

    bases: tuple[Base, ...] = ()
    departements: tuple[str, ...] = ()
    communes: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    near: tuple[float, float, float] | None = None
    """(latitude, longitude, km): entities placed within km of that point."""
    mode: FilterMode = FilterMode.BOOST
    ignored: tuple[str, ...] = ()
    """Values the model gave that mean nothing here (an unknown base), reported back to it."""

    @property
    def empty(self) -> bool:
        return not (
            self.bases
            or self.departements
            or self.communes
            or self.domains
            or self.periods
            or self.near
        )

    def describe(self) -> str:
        if self.empty:
            return "no filter"
        parts = [
            f"{name}={'|'.join(values)}"
            for name, values in (
                ("base", tuple(b.value for b in self.bases)),
                ("departement", self.departements),
                ("commune", self.communes),
                ("domain", self.domains),
                ("period", self.periods),
            )
            if values
        ]
        if self.near is not None:
            parts.append(f"near={self.near[0]:.4f},{self.near[1]:.4f} within {self.near[2]:g} km")
        return f"{', '.join(parts)} ({self.mode.value})"


def parse_filters(raw: object) -> Filters:
    """Filters from the model's JSON. Lenient: a lone string counts as a list of one."""
    if not isinstance(raw, dict):
        return Filters()
    given: dict[str, object] = {str(k): v for k, v in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    ignored: list[str] = []
    bases: list[Base] = []
    for value in _strings(given.get("base")):
        key = place_key(value).replace(" ", "-")
        match = next((b for b in Base if key in (b.value, b.value.replace("-", ""))), None)
        if match is None:
            ignored.append(f"base={value}")
        else:
            bases.append(match)
    mode_text = str(given.get("mode", "boost")).strip().lower()
    mode = FilterMode.HARD if mode_text == "hard" else FilterMode.BOOST
    return Filters(
        bases=tuple(dict.fromkeys(bases)),
        departements=_strings(given.get("departement")),
        communes=_strings(given.get("commune")),
        domains=_strings(given.get("domain")),
        periods=_strings(given.get("period")),
        near=_near(given.get("near")),
        mode=mode,
        ignored=tuple(ignored),
    )


#: Radius of a `near` filter when the model gives none, and the largest it may ask for.
NEAR_KM = 2.0
MAX_NEAR_KM = 50.0


def _near(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, dict):
        return None
    given: dict[str, object] = {str(k): v for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    lat = _number(given.get("lat", given.get("latitude")), np.nan)
    lon = _number(given.get("lon", given.get("longitude")), np.nan)
    if not (np.isfinite(lat) and np.isfinite(lon)) or abs(lat) > 90 or abs(lon) > 180:
        return None
    km = min(max(_number(given.get("km"), NEAR_KM), 0.1), MAX_NEAR_KM)
    return (lat, lon, km)


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    items: list[object] = list(value) if isinstance(value, list) else [value]  # pyright: ignore[reportUnknownArgumentType]
    return tuple(s for s in (str(v).strip() for v in items if v is not None) if s)


@dataclass(frozen=True, slots=True)
class Catalog:
    """The gallery's entities, in the column order of every score row."""

    infos: tuple[EntityInfo, ...]
    _column: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        ids = [i.entity_id for i in self.infos]
        if ids != sorted(ids):
            msg = "catalog entities must be sorted, as score columns are"
            raise ValueError(msg)
        object.__setattr__(self, "_column", {e: j for j, e in enumerate(ids)})

    @property
    def entity_ids(self) -> tuple[str, ...]:
        return tuple(i.entity_id for i in self.infos)

    def column(self, entity_id: str) -> int | None:
        return self._column.get(entity_id)

    def _near(self, near: tuple[float, float, float]) -> NDArray[np.bool_]:
        lat = np.asarray([i.position[0] if i.position else np.nan for i in self.infos])
        lon = np.asarray([i.position[1] if i.position else np.nan for i in self.infos])
        km = distance_km(near[0], near[1], lat, lon)
        return np.nan_to_num(km, nan=np.inf) <= near[2]

    def mask(self, filters: Filters) -> NDArray[np.bool_]:
        """Entities that match every given field."""
        keep = np.ones(len(self.infos), dtype=bool)
        if filters.empty:
            return keep
        bases = set(filters.bases)
        depts = {place_key(d) for d in filters.departements}
        communes = {place_key(c) for c in filters.communes}
        domains = set(tokens(" ".join(filters.domains)))
        periods = set(tokens(" ".join(filters.periods)))
        if filters.near is not None:
            keep &= self._near(filters.near)
        for j, info in enumerate(self.infos):
            if not keep[j]:
                continue
            keep[j] = (
                (not bases or info.base in bases)
                and (not depts or any(place_key(d) in depts for d in info.departements))
                and (not communes or any(place_key(c) in communes for c in info.communes))
                and (not domains or bool(domains & info.domain_words))
                and (not periods or bool(periods & info.period_words))
            )
        return keep


def apply_filters(scores: Floats, mask: NDArray[np.bool_], mode: FilterMode) -> Floats:
    """Scores after a filter: boosted where it matches, or cut where it does not."""
    out = scores.astype(np.float32, copy=True)
    if mode is FilterMode.HARD:
        out[~mask] = -np.inf
    else:
        out[mask & np.isfinite(out)] += BOOST
    return out


def matching_first(scores: Floats, mask: NDArray[np.bool_]) -> Floats:
    """A full ranking for a hard cut: the matching entities first, then the rest in order."""
    out = scores.astype(np.float32, copy=True)
    out[mask & np.isfinite(out)] += 10.0
    return out


# --- the world: gallery, queries, their scores -------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryData:
    """One query and its saved image scores (one value per catalog entity, -inf when hidden)."""

    set_id: str
    query: SearchQuery
    a7: Floats
    a3: Floats
    g5_weight: float
    """A7 weight G5 gives this photo: archive building 1.0, colour building 0.0, object 0.2."""
    bucket: str


@dataclass(frozen=True, slots=True)
class GalleryImages:
    """Gallery images with their entity's catalog column and both models' vectors."""

    image_ids: tuple[str, ...]
    columns: NDArray[np.int64]
    a7: NDArray[np.float16]
    a3: NDArray[np.float16]
    query_a7: Mapping[str, NDArray[np.float16]]
    """Each query photo's A7 vector (from the query run or the gallery run)."""
    query_a3: Mapping[str, NDArray[np.float16]]


class World:
    """Everything the tools search, shared by every query of a run."""

    def __init__(
        self,
        catalog: Catalog,
        text: TextGallery,
        images: GalleryImages,
        queries: Sequence[QueryData],
    ) -> None:
        names = np.asarray(catalog.entity_ids)
        text_names = np.unique(np.asarray(text.entity_ids))
        if not set(text_names.tolist()) <= set(catalog.entity_ids):
            msg = "the text gallery holds entities the catalog does not"
            raise ValueError(msg)
        self.catalog = catalog
        self.text = text
        self.images = images
        self.queries = {q.query.query_id: q for q in queries}
        codes = np.searchsorted(text_names, np.asarray(text.entity_ids))
        self._text_order = np.argsort(codes, kind="stable")
        self._text_starts = np.flatnonzero(np.r_[True, np.diff(codes[self._text_order]) != 0])
        self._text_columns = np.searchsorted(names, text_names)
        """Catalog column of each text entity. Entities with no record text (only Commons
        photos) score 0 in text searches."""
        self._text_row = {image_id: i for i, image_id in enumerate(text.image_ids)}
        self._image_row = {image_id: i for i, image_id in enumerate(images.image_ids)}

    def image_scores(self, q: QueryData, a7_weight: float) -> Floats:
        """A7 and A3 min-max scores mixed: `a7_weight` on A7, the rest on A3 (G1, G5)."""
        w = min(max(a7_weight, 0.0), 1.0)
        scaled = w * min_max(q.a7[None, :])[0] + (1 - w) * min_max(q.a3[None, :])[0]
        hidden = ~np.isfinite(q.a7) | ~np.isfinite(q.a3)
        return np.where(hidden, -np.inf, scaled).astype(np.float32)

    def g5(self, q: QueryData) -> Floats:
        return self.image_scores(q, q.g5_weight)

    def text_scores(self, q: QueryData, text: str) -> Floats:
        """BM25 of `text` per entity (best record), min-max scaled; hidden images do not count.

        All zero when no word matches.
        """
        per_image = self.text.index.scores(text)[self.text.record_rows].astype(np.float32)
        hidden = [
            self._text_row[i]
            for i in (q.query.image_id, *q.query.excluded_image_ids)
            if i in self._text_row
        ]
        per_image[hidden] = -np.inf
        per_entity = np.maximum.reduceat(per_image[self._text_order], self._text_starts)
        best = np.zeros(len(self.catalog.infos), dtype=np.float32)
        best[self._text_columns] = per_entity
        if not np.any(best[np.isfinite(best)] > 0):
            return np.where(np.isfinite(best), 0.0, -np.inf).astype(np.float32)
        return np.where(np.isfinite(best), min_max(best[None, :])[0], -np.inf).astype(np.float32)

    def best_images(self, q: QueryData, entity_ids: Sequence[str]) -> dict[str, str]:
        """For each entity, its gallery image closest to the query photo (never a hidden one)."""
        imgs = self.images
        cols = [self.catalog.column(e) for e in entity_ids]
        wanted = {c for c in cols if c is not None}
        rows = np.flatnonzero(np.isin(imgs.columns, list(wanted)))
        hidden = {
            self._image_row[i]
            for i in (q.query.image_id, *q.query.excluded_image_ids)
            if i in self._image_row
        }
        rows = np.asarray([r for r in rows if r not in hidden], dtype=np.int64)
        if rows.size == 0:
            return {}
        w = q.g5_weight
        a7 = np.asarray(imgs.a7[rows], dtype=np.float32) @ np.asarray(
            imgs.query_a7[q.query.image_id], dtype=np.float32
        )
        a3 = np.asarray(imgs.a3[rows], dtype=np.float32) @ np.asarray(
            imgs.query_a3[q.query.image_id], dtype=np.float32
        )
        sims = w * a7 + (1 - w) * a3
        best: dict[int, tuple[float, int]] = {}
        for row, sim in zip(rows.tolist(), sims.tolist(), strict=True):
            col = int(imgs.columns[row])
            if col not in best or sim > best[col][0]:
                best[col] = (sim, row)
        out: dict[str, str] = {}
        for entity_id, col in zip(entity_ids, cols, strict=True):
            if col is not None and col in best:
                out[entity_id] = imgs.image_ids[best[col][1]]
        return out

    def describe(self, entity_id: str, score: float | None = None) -> str:
        """ "<id> | <score> | <base> | <title> | <commune (département)>" for a result line."""
        col = self.catalog.column(entity_id)
        if col is None:
            return f"{entity_id} | unknown id"
        info = self.catalog.infos[col]
        shown = f" | {score:.2f}" if score is not None else ""
        return f"{entity_id}{shown} | {info.base.value} | {info.title} | {info.place}"


def top(scores: Floats, catalog: Catalog, k: int) -> list[tuple[str, float]]:
    """The k best finite entities, ties to the lower entity id (as in scoring)."""
    order = np.argsort(-scores, kind="stable")
    out: list[tuple[str, float]] = []
    for j in order[: max(k, 0)]:
        value = float(scores[j])
        if not np.isfinite(value):
            break
        out.append((catalog.infos[int(j)].entity_id, value))
    return out


def answer_scores(
    final: Sequence[str], fallback: Floats, catalog: Catalog, *, hidden: NDArray[np.bool_]
) -> Floats:
    """A full ranking: `final` first in its order, then every other entity by `fallback`.

    Hidden entities stay out, even if the model named them.
    """
    finite = np.where(np.isfinite(fallback), fallback, -1e6)
    out = finite.astype(np.float32)
    for i, entity_id in enumerate(dict.fromkeys(final)):  # an id listed twice keeps its first place
        col = catalog.column(entity_id)
        if col is not None and not hidden[col]:
            out[col] = 1e6 - i
    out[hidden] = -np.inf
    return out


# --- S1: one plan, carried out by code -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Plan:
    a7_weight: float = 0.5
    text_query: str = ""
    text_weight: float = 0.0
    filters: Filters = Filters()
    valid: bool = True
    """False when the answer held no JSON object; the plan is then G1 at 0.5, no text."""


def parse_plan(text: str) -> Plan:
    """The JSON plan of scenario S1. Missing parts fall back to the defaults of `Plan`."""
    raw = _json_object(text)
    if raw is None:
        return Plan(valid=False)
    image = raw.get("image_search")
    image_args: dict[str, object] = (
        {str(k): v for k, v in image.items()} if isinstance(image, dict) else {}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    )
    words = raw.get("text_search")
    word_args: dict[str, object] = (
        {str(k): v for k, v in words.items()} if isinstance(words, dict) else {}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    )
    query = str(word_args.get("query") or "").strip()
    return Plan(
        a7_weight=a7_weight_of(image_args),
        text_query=query,
        text_weight=_unit(word_args.get("weight"), 0.3) if query else 0.0,
        filters=parse_filters(raw.get("filters")),
    )


def a7_weight_of(args: Mapping[str, object]) -> float:
    """`model` "A7" is 1, "A3" is 0, "mix" takes `a7_weight` (0.5 when missing)."""
    model = str(args.get("model", "mix")).strip().upper()
    if model == "A7":
        return 1.0
    if model == "A3":
        return 0.0
    return _unit(args.get("a7_weight"), 0.5)


def _unit(value: object, default: float) -> float:
    try:
        number = float(str(value))
    except ValueError:
        return default
    if not np.isfinite(number):
        return default
    return min(max(number, 0.0), 1.0)


def _json_object(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        raw: object = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return {str(k): v for k, v in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def plan_scores(world: World, q: QueryData, plan: Plan) -> tuple[Floats, Floats]:
    """The plan's mixed scores before filters, and its full final ranking after them."""
    mixed = world.image_scores(q, plan.a7_weight)
    if plan.text_query and plan.text_weight > 0:
        words = world.text_scores(q, plan.text_query)
        mixed = ((1 - plan.text_weight) * mixed + plan.text_weight * words).astype(np.float32)
    if plan.filters.empty:
        return mixed, mixed
    mask = world.catalog.mask(plan.filters)
    if plan.filters.mode is FilterMode.HARD:
        return mixed, matching_first(mixed, mask)
    return mixed, apply_filters(mixed, mask, FilterMode.BOOST)


def parse_ranking(text: str, key: str = "ranking") -> tuple[str, ...]:
    """Entity ids listed under `key` in a JSON answer (S2's review); empty when none."""
    raw = _json_object(text)
    if raw is None:
        return ()
    return _strings(raw.get(key))


# --- S3: one query's tool calls ------------------------------------------------------------------


class Tool(StrEnum):
    IMAGE_SEARCH = "image_search"
    TEXT_SEARCH = "text_search"
    VIEW_RESULTS = "view_results"
    FINAL_ANSWER = "final_answer"


@dataclass(frozen=True, slots=True)
class ToolResult:
    text: str
    """What the model reads back."""
    listed: tuple[str, ...] = ()
    """Entities a search listed, in order."""
    thumbnails: tuple[tuple[str, str, str], ...] = ()
    """(label, entity id, image id) to show the model after the tool answers."""
    final: tuple[str, ...] | None = None
    filters: Filters | None = None
    error: bool = False


class Session:
    """One query's tools: runs each call, remembers what the model has seen and last listed."""

    def __init__(self, world: World, q: QueryData) -> None:
        self.world = world
        self.q = q
        self.seen: set[str] = set()
        self.last_list: tuple[str, ...] = ()

    def run(self, name: str, arguments: str) -> ToolResult:
        given = _arguments(arguments)
        if isinstance(given, ToolResult):
            return given
        match name:
            case Tool.IMAGE_SEARCH:
                return self.image_search(given)
            case Tool.TEXT_SEARCH:
                return self.text_search(given)
            case Tool.VIEW_RESULTS:
                return self.view_results(_strings(given.get("entity_ids")))
            case Tool.FINAL_ANSWER:
                return self.final_answer(_strings(given.get("entity_ids")))
            case _:
                return ToolResult(f"error: no tool named {name!r}", error=True)

    def image_search(self, args: Mapping[str, object]) -> ToolResult:
        weight = a7_weight_of(args)
        label = "A7" if weight == 1 else "A3" if weight == 0 else f"A7 {weight:.2f} + A3"
        return self._search(f"image_search {label}", self.world.image_scores(self.q, weight), args)

    def text_search(self, args: Mapping[str, object]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult("error: text_search needs a query", error=True)
        scores = self.world.text_scores(self.q, query)
        if not np.any(scores[np.isfinite(scores)] > 0):
            return ToolResult(f"text_search {query!r}: no record holds these words", listed=())
        return self._search(f"text_search {query!r}", scores, args)

    def _search(self, title: str, scores: Floats, args: Mapping[str, object]) -> ToolResult:
        filters = parse_filters(args.get("filters"))
        k = min(max(int(_number(args.get("top_k"), DEFAULT_TOP_K)), 1), MAX_TOP_K)
        catalog = self.world.catalog
        if not filters.empty:
            mask = catalog.mask(filters)
            scores = apply_filters(scores, mask, filters.mode)
            kept = int((mask & np.isfinite(scores)).sum())
            title += f", filters {filters.describe()}: {kept} entities match"
        if filters.ignored:
            title += f" (ignored: {', '.join(filters.ignored)})"
        found = top(scores, catalog, k)
        lines = [f"{i}. {self.world.describe(e, s)}" for i, (e, s) in enumerate(found, start=1)]
        listed = tuple(e for e, _ in found)
        self.seen.update(listed)
        self.last_list = listed or self.last_list
        body = "\n".join(lines) if lines else "nothing found"
        return ToolResult(f"{title}\n{body}", listed=listed, filters=filters)

    def view_results(self, entity_ids: Sequence[str]) -> ToolResult:
        if not entity_ids:
            return ToolResult("error: view_results needs entity_ids", error=True)
        asked = list(dict.fromkeys(entity_ids))[:MAX_VIEW]
        known = [e for e in asked if self.world.catalog.column(e) is not None]
        images = self.world.best_images(self.q, known)
        thumbs: list[tuple[str, str, str]] = []
        lines: list[str] = []
        for i, entity_id in enumerate(asked, start=1):
            if entity_id not in images:
                lines.append(f"[{i}] {entity_id}: unknown id, no image")
                continue
            lines.append(f"[{i}] {self.world.describe(entity_id)}")
            thumbs.append((f"[{i}] {entity_id}", entity_id, images[entity_id]))
        self.seen.update(e for _, e, _ in thumbs)
        note = f" (only the first {MAX_VIEW})" if len(entity_ids) > MAX_VIEW else ""
        text = f"view_results{note}: thumbnails follow in the next message\n" + "\n".join(lines)
        return ToolResult(text, thumbnails=tuple(thumbs))

    def final_answer(self, entity_ids: Sequence[str]) -> ToolResult:
        kept = tuple(e for e in dict.fromkeys(entity_ids) if e in self.seen)[:MAX_ANSWER]
        dropped = len(set(entity_ids)) - len(kept)
        text = f"final answer: {len(kept)} entities" + (
            f"; {dropped} left out (never seen in a result)" if dropped else ""
        )
        return ToolResult(text, final=kept)


def _arguments(arguments: str) -> dict[str, object] | ToolResult:
    try:
        args: object = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return ToolResult(f"error: the arguments are not JSON: {arguments[:200]}", error=True)
    if not isinstance(args, dict):
        return ToolResult("error: the arguments must be a JSON object", error=True)
    return {str(k): v for k, v in args.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def _number(value: object, default: float) -> float:
    try:
        number = float(str(value))
    except ValueError:
        return default
    return number if np.isfinite(number) else default
