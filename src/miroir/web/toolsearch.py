"""The searches the conversation page's model calls (step H5), run where the engine runs.

Step H4 gave a vision model tools over saved vectors. Here the same tools search the live
index with a visitor's new photo, so they run beside the models: in the GPU container when
deployed (`miroir.web.modal_app.Search.tool_search`), with fake models on the Mac.

- image search: the photo's cosines from both models (`Engine.photo_cosines`), mixed per
  entity as the site mixes them (`ranking.fuse`), at G5's routed weight or the one asked for;
- text search: BM25 over record text, as H4's `text_search`;
- view: each entity's images, closest to the photo first, so the page can show the model
  thumbnails (it checks which may go out, `miroir.web.publishable`);
- like: the records that look like one the model has already seen, by that record's own
  stored vectors (`Engine.stored_cosines`): no model runs, so it costs no embedding.

Filters favour by default: a matching entity gains `BOOST` on its 0-1 score, and a position
adds the soft prior of step H2. A hard cut happens only when the caller asks (H4's 10-photo
test: a hard filter on a wrong guess lost the answer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from miroir.eval.bm25 import tokens
from miroir.images.mount import crop_picture, find_picture
from miroir.records.bases import Base, base_of, place_key
from miroir.records.model import Collection
from miroir.web import facets, ranking
from miroir.web.catalog import CARD_SIDE, COLLECTION_LABELS, MATCHED_LABELS, display_url
from miroir.web.engine import Engine, weighted
from miroir.web.nearby import Position

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]
Bools = NDArray[np.bool_]

#: Added to an entity's 0-1 score when it matches a soft filter (H4's value).
BOOST = 0.25
#: Most entities one search lists, and the default.
MAX_TOP_K = 30
DEFAULT_TOP_K = 10
#: Images of one entity offered for the model to look at, closest first.
MODEL_IMAGES = 4
#: Side asked of the image service for a thumbnail the model sees.
THUMB_SIDE = 256
#: Words too common in period fields to tell records apart.
_PERIOD_NOISE = frozenset({"siecle", "quart", "moitie", "milieu", "debut", "fin", "annee"})


class SearchKind(StrEnum):
    IMAGE = "image"
    TEXT = "text"
    VIEW = "view"
    LIKE = "like"


@dataclass(frozen=True, slots=True)
class Near:
    latitude: float
    longitude: float
    km: float


@dataclass(frozen=True, slots=True)
class ToolFilters:
    """Every field given must match one of its values; an empty field matches all."""

    bases: tuple[Base, ...] = ()
    departements: tuple[str, ...] = ()
    communes: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    near: Near | None = None
    hard: bool = False
    """Keep only matching entities. False (the default): favour them."""

    @property
    def empty(self) -> bool:
        return self.near is None and not self.fields

    @property
    def fields(self) -> bool:
        """Some field other than `near` is given."""
        return bool(self.bases or self.departements or self.communes or self.domains) or bool(
            self.periods
        )


@dataclass(frozen=True, slots=True)
class ToolQuery:
    kind: SearchKind
    image: bytes | None = None
    """The visitor's photo, or the part of it they framed. Image search and view."""
    text: str = ""
    a7_weight: float | None = None
    """EmbeddingGemma 2's share, MegaLoc taking the rest; None: G5's routed weight."""
    filters: ToolFilters = field(default_factory=ToolFilters)
    top_k: int = DEFAULT_TOP_K
    entity_ids: tuple[str, ...] = ()
    """View: the entities to show. Like: the one entity to start from."""
    image_id: str | None = None
    """Like: the entity's image to start from (the one the model saw); None: its cover."""


@dataclass(frozen=True, slots=True)
class ModelImage:
    image_id: str
    url: str


@dataclass(frozen=True, slots=True)
class Hit:
    """One entity a tool found, with what a result line and a result card need."""

    entity_id: str
    score: float
    base: str
    title: str
    place: str
    authors: str
    period: str
    collection: str
    collection_label: str
    pop_url: str
    image_url: str | None
    image_fallback_url: str | None
    matched: str | None
    photo_count: int
    images: tuple[ModelImage, ...]
    """Its images the model may be shown, closest to the photo first (the page checks which
    may leave)."""
    matches: bool = False
    """It matches the filters."""


@dataclass(frozen=True, slots=True)
class ToolHits:
    hits: tuple[Hit, ...]
    matching: int | None = None
    """Entities matching the filters, when there were any."""
    a7_weight: float | None = None
    bucket: str | None = None
    words_found: bool | None = None
    mount: bool = False
    """The photo is an archive print on its card: the print alone was searched."""
    near_ignored: bool = False
    """A position was given but the index knows no places."""


@dataclass(frozen=True, slots=True)
class _Facts:
    """What filters read, per entity (catalog entity order)."""

    bases: tuple[Base | None, ...]
    communes: tuple[frozenset[str], ...]
    departements: tuple[frozenset[str], ...]
    domain_words: tuple[frozenset[str], ...]
    period_words: tuple[frozenset[str], ...]
    periods: tuple[frozenset[str], ...]
    """Broad periods (`facets.PERIODS`)."""


class ToolSearch:
    """The tools' searches over one engine."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._facts: _Facts | None = None

    def run(self, query: ToolQuery) -> ToolHits:
        match query.kind:
            case SearchKind.IMAGE:
                return self._image(query)
            case SearchKind.TEXT:
                return self._text(query)
            case SearchKind.VIEW:
                return self._view(query)
            case SearchKind.LIKE:
                return self._like(query)

    # --- searches ------------------------------------------------------------------------------

    def _image(self, query: ToolQuery) -> ToolHits:
        if query.image is None:
            msg = "an image search needs a photo"
            raise ValueError(msg)
        photo, mount = _print_alone(query.image)
        cosines = self.engine.photo_cosines(photo)
        w = cosines.bucket.a7_weight if query.a7_weight is None else query.a7_weight
        w = min(max(w, 0.0), 1.0)
        catalog = self.engine.catalog
        signals = [
            ranking.Signal(ranking.entity_max(cosines.a7, catalog), w),
            ranking.Signal(ranking.entity_max(cosines.a3, catalog), 1 - w),
        ]
        fused = ranking.fuse([s for s in signals if s.weight > 0])
        per_image = weighted(cosines.a7, cosines.a3, w)
        found = self._ranked(fused, per_image, query)
        return ToolHits(
            found.hits,
            found.matching,
            a7_weight=w,
            bucket=cosines.bucket.name,
            mount=mount,
            near_ignored=found.near_ignored,
        )

    def _text(self, query: ToolQuery) -> ToolHits:
        text = query.text.strip()
        if not text:
            msg = "a text search needs words"
            raise ValueError(msg)
        catalog = self.engine.catalog
        per_record = self.engine.bm25.scores(text).astype(np.float32)
        if not bool((per_record > 0).any()):
            return ToolHits((), words_found=False)
        records = catalog.image_record
        per_image = np.where(records >= 0, per_record[np.maximum(records, 0)], 0.0).astype(
            np.float32
        )
        # Only entities holding a word are listed.
        best = ranking.entity_max(per_image, catalog)
        fused = ranking.min_max(np.where(best > 0, best, -np.inf).astype(np.float32))
        found = self._ranked(fused, per_image, query)
        return ToolHits(
            found.hits, found.matching, words_found=True, near_ignored=found.near_ignored
        )

    def _like(self, query: ToolQuery) -> ToolHits:
        catalog = self.engine.catalog
        entity = catalog.entity_row.get(query.entity_ids[0]) if query.entity_ids else None
        if entity is None:
            msg = "a search by a record's image needs a known entity"
            raise ValueError(msg)
        row = self.start_row(entity, query.image_id)
        a7, a3 = self.engine.stored_cosines(row)
        w = 0.5 if query.a7_weight is None else min(max(query.a7_weight, 0.0), 1.0)
        signals = [
            ranking.Signal(ranking.entity_max(a7, catalog), w),
            ranking.Signal(ranking.entity_max(a3, catalog), 1 - w),
        ]
        fused = ranking.fuse([s for s in signals if s.weight > 0])
        fused[entity] = -np.inf  # the record itself is not news
        found = self._ranked(fused, weighted(a7, a3, w), query)
        return ToolHits(found.hits, found.matching, a7_weight=w, near_ignored=found.near_ignored)

    def start_row(self, entity: int, image_id: str | None) -> int:
        """The image a search by look starts from: the one the model saw when it belongs to
        the entity, else its cover (POP images before Commons photos)."""
        catalog = self.engine.catalog
        rows = [int(r) for r in catalog.entity_image_rows(entity)]
        if image_id is not None:
            row = catalog.image_row.get(image_id)
            if row is not None and row in rows:
                return row
        return min(
            rows,
            key=lambda r: (catalog.image_record[r] < 0, not catalog.images[r].is_cover, r),
        )

    def _view(self, query: ToolQuery) -> ToolHits:
        catalog = self.engine.catalog
        per_image: Floats | None = None
        if query.image is not None:
            photo, _ = _print_alone(query.image)
            cosines = self.engine.photo_cosines(photo)
            per_image = weighted(cosines.a7, cosines.a3, cosines.bucket.a7_weight)
        hits: list[Hit] = []
        for entity_id in dict.fromkeys(query.entity_ids):
            row = catalog.entity_row.get(entity_id)
            if row is None:
                continue
            hit = self._hit(row, per_image, 0.0)
            if hit is not None:
                hits.append(hit)
        return ToolHits(tuple(hits))

    # --- ranking and filters ---------------------------------------------------------------------

    def _ranked(self, fused: Floats, per_image: Floats, query: ToolQuery) -> _Ranked:
        filters = query.filters
        scores = fused.astype(np.float32, copy=True)
        mask = self.mask(filters)
        near_ignored = False
        prior: Floats | None = None
        if filters.near is not None:
            places = self.engine.places
            if places is None:
                near_ignored = True
            else:
                position = Position(filters.near.latitude, filters.near.longitude)
                if filters.hard:
                    mask &= places.near(position, filters.near.km)
                else:
                    # H2's soft prior: its 1 km scale for the default 2 km radius.
                    prior = places.prior(position, scale_km=max(filters.near.km / 2, 0.1))
        alive = np.isfinite(scores)
        if filters.hard:
            scores[~mask] = -np.inf
        else:
            if filters.fields:
                scores[mask & alive] += BOOST
            if prior is not None:
                scores = (scores + prior).astype(np.float32)
        top = ranking.top_entities(scores, min(max(query.top_k, 1), MAX_TOP_K))
        hits = [
            hit
            for row in top
            if (hit := self._hit(int(row), per_image, float(scores[row]), matches=bool(mask[row])))
        ]
        matching = None if filters.empty else int((mask & alive).sum())
        return _Ranked(tuple(hits), matching, near_ignored)

    def mask(self, filters: ToolFilters) -> Bools:
        """Entities that match every given field (`near` aside)."""
        facts = self._entity_facts()
        n = self.engine.catalog.entity_count
        keep = np.ones(n, dtype=bool)
        if not filters.fields:
            return keep
        bases = set(filters.bases)
        depts = {place_key(d) for d in filters.departements}
        communes = {place_key(c) for c in filters.communes}
        domains = set(tokens(" ".join(filters.domains)))
        broad = set(facets.periods(filters.periods))
        period_words = set(tokens(" ".join(filters.periods))) - _PERIOD_NOISE
        for j in range(n):
            keep[j] = (
                (not bases or facts.bases[j] in bases)
                and (not depts or bool(depts & facts.departements[j]))
                and (not communes or bool(communes & facts.communes[j]))
                and (not domains or bool(domains & facts.domain_words[j]))
                and (
                    not filters.periods
                    or bool(broad & facts.periods[j])
                    or bool(period_words & facts.period_words[j])
                )
            )
        return keep

    def _entity_facts(self) -> _Facts:
        if self._facts is not None:
            return self._facts
        catalog = self.engine.catalog
        n = catalog.entity_count
        bases: list[Base | None] = [None] * n
        communes: list[set[str]] = [set() for _ in range(n)]
        depts: list[set[str]] = [set() for _ in range(n)]
        domains: list[set[str]] = [set() for _ in range(n)]
        period_words: list[set[str]] = [set() for _ in range(n)]
        periods: list[set[str]] = [set() for _ in range(n)]
        for record, row in zip(catalog.records, map(int, catalog.record_entity), strict=True):
            if bases[row] is None:
                bases[row] = base_of(Collection(record.collection), record.entity_id)
            if record.commune:
                communes[row].add(place_key(record.commune))
            if record.departement:
                depts[row].add(place_key(record.departement))
            domains[row].update(tokens(" ".join((*record.domain, *record.denomination))))
            period_words[row].update(tokens(" ".join(record.period)))
            periods[row].update(facets.periods(record.period))
        self._facts = _Facts(
            bases=tuple(bases),
            communes=tuple(frozenset(s) for s in communes),
            departements=tuple(frozenset(s) for s in depts),
            domain_words=tuple(frozenset(s) for s in domains),
            period_words=tuple(frozenset(s) for s in period_words),
            periods=tuple(frozenset(s) for s in periods),
        )
        return self._facts

    # --- one entity ---------------------------------------------------------------------------

    def _hit(
        self, entity: int, per_image: Floats | None, score: float, *, matches: bool = False
    ) -> Hit | None:
        catalog = self.engine.catalog
        rows = catalog.entity_image_rows(entity)
        records = catalog.entity_records(entity)
        if not records:
            return None
        if per_image is None:
            ordered = sorted(
                (int(r) for r in rows), key=lambda r: (not catalog.images[r].is_cover, r)
            )
        else:
            ordered = sorted((int(r) for r in rows), key=lambda r: (-float(per_image[r]), r))
        pop = [r for r in ordered if catalog.image_record[r] >= 0]
        shown = catalog.images[pop[0]] if pop else None
        record = (catalog.record_of(pop[0]) if pop else None) or records[0]
        authors = list(dict.fromkeys(a for r in records for a in r.authors))
        periods = list(dict.fromkeys(p for r in records for p in r.period))
        matched = None
        if shown is not None and not shown.is_cover:
            matched = MATCHED_LABELS.get(shown.image_type, "Autre vue")
        return Hit(
            entity_id=catalog.entity_ids[entity],
            score=round(score, 4),
            base=base_of(Collection(record.collection), record.entity_id).value,
            title=record.display_title,
            place=record.place,
            authors=", ".join(authors[:2]),
            period=", ".join(periods[:2]),
            collection=record.collection,
            collection_label=COLLECTION_LABELS.get(record.collection, record.collection),
            pop_url=record.pop_url,
            image_url=None if shown is None else display_url(shown, CARD_SIDE),
            image_fallback_url=None if shown is None else shown.url,
            matched=matched,
            photo_count=len(pop),
            images=tuple(
                ModelImage(catalog.images[r].image_id, display_url(catalog.images[r], THUMB_SIDE))
                for r in ordered[:MODEL_IMAGES]
            ),
            matches=matches,
        )


@dataclass(frozen=True, slots=True)
class _Ranked:
    hits: tuple[Hit, ...]
    matching: int | None
    near_ignored: bool


def _print_alone(data: bytes) -> tuple[bytes, bool]:
    """The print alone when the photo shows an archive print on its card (as the site does)."""
    try:
        picture = find_picture(data)
    except OSError as err:
        msg = f"cannot read the photo: {err}"
        raise ValueError(msg) from err
    if picture is None:
        return data, False
    return crop_picture(data, picture), True
