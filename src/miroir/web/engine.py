"""One search: a photo and/or words, plus filters, in; cards, each with its level, out.

`Engine` holds the catalog, the routing and the gallery vectors. The models sit behind two
small interfaces so the same engine runs in the Modal GPU container (`miroir.remote.search`)
and on the Mac with fake models (`miroir.web.fake`, tests, `miroir web dev`).
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageOps

from miroir.caption import CaptionPlace, Gazetteer, clean_caption
from miroir.eval.bm25 import Bm25
from miroir.images.colour import chroma_spread
from miroir.images.fetch import JPEG_QUALITY, LONG_SIDE
from miroir.images.mount import Picture, crop_picture, find_picture
from miroir.web import ranking
from miroir.web.archive import SITE_ARCHIVE, ArchiveRule, CaptionReader, gazetteer
from miroir.web.catalog import (
    CARD_SIDE,
    CATALOG_FILE,
    COLLECTION_LABELS,
    MATCHED_LABELS,
    Catalog,
    CatalogRecord,
    Filters,
    display_url,
    read_catalog,
)
from miroir.web.levels import (
    LEVEL_LABELS,
    LEVELS_FILE,
    WORDS_ROUTE,
    Level,
    Levels,
    gap_of,
    read_levels,
)
from miroir.web.nearby import NEAR_KM, PLACES_FILE, EntityPlaces, Position, read_places
from miroir.web.routing import (
    GUESS_FILE,
    ROUTING_FILE,
    Bucket,
    Guess,
    PhotoType,
    Routing,
    TextWeights,
    read_guess,
    read_routing,
)
from miroir.web.store import ExactStore, read_exact, shortlist_cosines
from miroir.web.sure import SURE_FILE, SureRule, read_sure

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]

#: Cards per page of results.
RESULTS = 24
#: Longest text query kept, in characters.
MAX_TEXT = 300
#: Cards the first card's gap is measured over (`miroir.web.levels`).
GAP_OVER = 30
A7_FILE = "a7.npy"
A3_FILE = "a3.npy"
#: Optional: MegaLoc vectors stored reduced (PCA). Holds `mean` and `components`.
A3_PROJECTION_FILE = "a3_projection.npz"


class Encoders(Protocol):
    """The two models, as the gallery was embedded with them."""

    def image(self, jpeg: bytes) -> tuple[Floats, Floats]:
        """(EmbeddingGemma 2, MegaLoc) vectors of a photo, before any projection."""
        ...

    def text(self, text: str) -> Floats:
        """EmbeddingGemma 2 vector of a search query."""
        ...


class GalleryVectors(Protocol):
    def a7(self, query: Floats) -> Floats:
        """Cosine of the query with every image's EmbeddingGemma 2 vector, in catalog order."""
        ...

    def a3(self, query: Floats) -> Floats:
        """The same for MegaLoc (after the projection, when the index has one)."""
        ...

    def a7_row(self, row: int) -> Floats:
        """One image's stored EmbeddingGemma 2 vector (no model runs)."""
        ...

    def a3_row(self, row: int) -> Floats:
        """One image's stored MegaLoc vector, in the space `a3` searches."""
        ...


def weighted(a7: Floats, a3: Floats, a7_weight: float) -> Floats:
    """`a7_weight * a7 + (1 - a7_weight) * a3`, leaving out a term weighted 0 (so a MegaLoc
    score of -inf, off the shortlist, cannot turn an EmbeddingGemma 2-only score into NaN)."""
    if a7_weight >= 1:
        return np.asarray(a7, dtype=np.float32)
    if a7_weight <= 0:
        return np.asarray(a3, dtype=np.float32)
    return (a7_weight * a7 + (1 - a7_weight) * a3).astype(np.float32)


@dataclass(frozen=True, slots=True)
class Projection:
    """A linear map applied to MegaLoc query vectors, when the index stores them reduced."""

    mean: Floats
    components: Floats
    """Shape (reduced dim, 8448)."""

    def apply(self, vector: Floats) -> Floats:
        return unit(((vector - self.mean) @ self.components.T).astype(np.float32))


@dataclass(frozen=True, slots=True)
class SearchRequest:
    image: bytes | None = None
    """The photo (or the part of it the person boxed), any format Pillow reads."""
    text: str = ""
    filters: Filters = field(default_factory=Filters)
    limit: int = RESULTS
    position: Position | None = None
    """Where the person stands, when they chose to share it (step H2). Used, never kept."""
    mount: bool = True
    """Look for an archive mount around the photo and search the print alone."""
    caption: str | None = None
    """The mount's caption as the person corrected it ("" when they removed it). None: read
    it from the photo when it has a mount."""
    exclude_images: tuple[str, ...] = ()
    """Measurement only (not in the API): images taken out of the gallery, such as a test
    query's own shoot."""


@dataclass(frozen=True, slots=True)
class Card:
    """One building or object (functional spec §6)."""

    entity_id: str
    title: str
    place: str
    collection: str
    collection_label: str
    pop_url: str
    image_url: str | None
    """The entity's POP image that best matches the query. A Commons photo may have matched
    best: it helps the search but is never shown. None when the entity has no POP image."""
    image_fallback_url: str | None
    """The same file straight from the bucket, if the first URL fails."""
    matched: str | None
    """How the image shown differs from the cover ("Détail", "Revers"...), or None."""
    photo_count: int
    """Its POP images."""
    score: float
    level: Level
    level_label: str
    """How close it is, in the visitor's words ("Très proche", "Proche", "Ressemblance")."""


@dataclass(frozen=True, slots=True)
class SearchResponse:
    mode: str
    """"photo", "text" or "photo+text"."""
    confidence: float | None
    """Photo searches: the first card's routed mix of raw cosines. None for words only."""
    guessed_type: str | None
    bucket: str | None
    chroma: float | None
    """The photo's chroma spread (`images.colour`): near 0 for black and white."""
    cards: tuple[Card, ...]
    timings_ms: dict[str, float] = field(default_factory=dict[str, float])
    nearby: int | None = None
    """With a position: entities known within `near_km` of it (the near ones rank higher)."""
    near_km: float | None = None
    words_found: bool | None = None
    """Searches with words: whether any record holds one of them. When none does, the
    cards come from the text vector alone, and the page says so."""
    mount: bool = False
    """The photo shows an archive print on its card (`images.mount`): its caption was read."""
    picture: tuple[float, float, float, float] | None = None
    """The print found inside an archive mount (left, top, right, bottom, in shares of the
    photo), which was searched instead of the whole photo."""
    caption: str | None = None
    """The mount's caption the search used, read or as the person gave it."""
    caption_read: bool = False
    """The caption was read from the photo in this search (not given by the person)."""
    caption_place: str | None = None
    """The commune the caption names, as the index writes it; buildings near it rank higher."""


class Engine:
    def __init__(
        self,
        *,
        catalog: Catalog,
        routing: Routing,
        guess: Guess,
        encoders: Encoders,
        gallery: GalleryVectors,
        projection: Projection | None = None,
        exact: ExactStore | None = None,
        places: EntityPlaces | None = None,
        sure: SureRule | None = None,
        levels: Levels | None = None,
        captions: CaptionReader | None = None,
        archive: ArchiveRule = SITE_ARCHIVE,
    ) -> None:
        self.catalog = catalog
        self.places = places
        self.routing = routing
        self.guess = guess
        self.encoders = encoders
        self.gallery = gallery
        self.projection = projection
        self.exact = exact
        """MegaLoc whole in host memory (`miroir.web.store`): the GPU then holds it reduced,
        only to shortlist images, and the shortlisted ones are scored exactly from here."""
        self.sure = sure
        """The older rule a bundle without `levels.json` falls back on (objects)."""
        self.levels = levels
        self.captions = captions
        self.archive = archive
        self.gazetteer: Gazetteer | None = None if places is None else gazetteer(catalog, places)
        start = time.monotonic()
        self.bm25 = Bm25([record_text(r) for r in catalog.records])
        logger.info(
            "BM25 over %d records in %.1f s", len(catalog.records), time.monotonic() - start
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        start = time.monotonic()
        text = request.text.strip()[:MAX_TEXT]
        if request.image is None and not text:
            msg = "a search needs a photo or words"
            raise ValueError(msg)
        timings: dict[str, float] = {}
        mask = self.catalog.image_mask(request.filters)
        if request.exclude_images:
            row = self.catalog.image_row
            mask = mask.copy()
            mask[[row[i] for i in request.exclude_images if i in row]] = False
        signals: list[ranking.Signal] = []
        per_image: Floats | None = None
        photo: _Photo | None = None
        words_found: bool | None = None
        archive = _Archive()
        if request.image is not None:
            archive = self._archive(request, timings)
            photo = self._photo(archive.query or request.image, mask, timings)
            signals += photo.signals
            per_image = photo.per_image
        if archive.caption and self.archive.words > 0:
            weights = self.routing.text.for_caption()
            words = self._words(archive.caption, mask, timings, weights, key="caption")
            signals += [
                ranking.Signal(s.per_entity, s.weight * self.archive.words) for s in words.signals
            ]
        if text:
            words = self._words(text, mask, timings, self.routing.text)
            words_found = words.found
            scale = 1.0 if photo is None else self.routing.text.refine
            signals += [ranking.Signal(s.per_entity, s.weight * scale) for s in words.signals]
            if per_image is None:
                per_image = words.per_image
        assert per_image is not None
        tick = time.monotonic()
        unplaced = ranking.fuse(signals)
        place = archive.place if self.archive.place_bonus > 0 else None
        fused, nearby = self._near(unplaced, place, request.position)
        top = ranking.top_entities(fused, max(1, min(request.limit, 100)))
        levels = self._levels(photo, unplaced, fused, top)
        cards = tuple(
            card
            for row, level in zip(top, levels, strict=True)
            if (card := self._card(int(row), per_image, mask, float(fused[row]), level)) is not None
        )
        confidence = None
        if photo is not None and len(top):
            confidence = photo.confidence(int(top[0]))
        timings["rank"] = _ms(tick)
        timings["total"] = _ms(start)
        return SearchResponse(
            mode="photo+text" if photo and text else ("photo" if photo else "text"),
            confidence=confidence,
            guessed_type=None if photo is None else photo.photo_type.value,
            bucket=None if photo is None else photo.bucket,
            chroma=None if photo is None else photo.chroma,
            cards=cards,
            timings_ms=timings,
            nearby=nearby,
            near_km=None if nearby is None else NEAR_KM,
            words_found=words_found,
            mount=archive.mount,
            picture=archive.box,
            caption=archive.caption or None,
            caption_read=archive.read,
            caption_place=None if place is None else place.label,
        )

    def _levels(
        self, photo: _Photo | None, unplaced: Floats, fused: Floats, top: ranking.Ints
    ) -> list[Level]:
        """Each card's level (`miroir.web.levels`). A photo search reads the photo's routed
        cosines, whatever words refine it; words alone read the fused score, before any
        position bonus."""
        if not len(top):
            return []
        head = ranking.top_entities(fused, GAP_OVER)
        if photo is not None:
            route = photo.bucket
            over = [photo.confidence(int(r)) for r in head]
        else:
            route = WORDS_ROUTE
            over = [float(unplaced[r]) for r in head]
        rule = None if self.levels is None else self.levels.for_route(route)
        if rule is not None:
            return rule.card_levels(len(top), over)
        # No levels for this route in the bundle: the older "sure" rule, on the first card.
        sure = False
        if photo is not None:
            sure = over[0] >= photo.cutoff
            if self.sure is not None and photo.photo_type is self.sure.photo_type:
                sure = gap_of(over[: self.sure.over]) >= self.sure.margin
        first = Level.VERY_CLOSE if sure else Level.LIKENESS
        return [first, *[Level.LIKENESS] * (len(top) - 1)]

    def _near(
        self, fused: Floats, place: CaptionPlace | None, position: Position | None
    ) -> tuple[Floats, int | None]:
        """Step H2's soft prior around the caption's commune, else around the person's
        position; and how many entities lie near the person. -inf (filtered out) stays."""
        if self.places is None:
            return fused, None
        if place is not None:
            # A print's caption says where the building stands; the person's position does not.
            around = Position(place.latitude, place.longitude)
            rule = self.archive
            prior = self.places.prior(around, bonus=rule.place_bonus, scale_km=rule.place_scale_km)
            return fused + prior, None
        if position is not None:
            return fused + self.places.prior(position), int(self.places.near(position).sum())
        return fused, None

    # --- archive prints ----------------------------------------------------------------------

    def _archive(self, request: SearchRequest, timings: dict[str, float]) -> _Archive:
        """The print inside a mount, and its caption (read only when there is a mount and the
        person gave none)."""
        assert request.image is not None
        tick = time.monotonic()
        try:
            picture = find_picture(request.image) if request.mount else None
        except (OSError, PILImage.DecompressionBombError) as err:
            msg = f"cannot read the photo: {err}"
            raise ValueError(msg) from err
        timings["mount"] = _ms(tick)
        query = crop_picture(request.image, picture) if picture and self.archive.crop else None
        read = False
        if request.caption is not None:
            caption = clean_caption(request.caption)
        elif picture is not None and self.captions is not None:
            tick = time.monotonic()
            caption = clean_caption(self.captions.read(request.image))
            timings["caption"] = _ms(tick)
            read = True
        else:
            caption = ""
        place = self.gazetteer.find(caption) if caption and self.gazetteer else None
        return _Archive(
            mount=picture is not None,
            query=query,
            box=None if query is None or picture is None else _shares(picture),
            caption=caption,
            read=read,
            place=place,
        )

    # --- the two kinds of query ------------------------------------------------------------

    def photo_cosines(self, data: bytes) -> PhotoCosines:
        """A photo's cosines with every image, both models, and the bucket G5 routes it to.
        The conversation page's image search mixes them its own way (`miroir.web.toolsearch`)."""
        mask = np.ones(len(self.catalog.images), dtype=bool)
        return self._cosines(prepare_photo(data), mask, {})

    def stored_cosines(self, row: int) -> tuple[Floats, Floats]:
        """(EmbeddingGemma 2, MegaLoc) cosines of one gallery image's stored vectors with every
        image: a search by a record's own image, with no model run (the conversation page's
        `more_like`)."""
        a7 = self.gallery.a7(unit(self.gallery.a7_row(row)))
        a3 = self.gallery.a3(unit(self.gallery.a3_row(row)))
        if self.exact is not None:
            full = unit(np.asarray(self.exact.rows[row], dtype=np.float32) * self.exact.scale[row])
            a3 = shortlist_cosines(self.exact, full, a7, a3)
        return a7.astype(np.float32), a3.astype(np.float32)

    def _cosines(
        self, jpeg: bytes, mask: NDArray[np.bool_], timings: dict[str, float]
    ) -> PhotoCosines:
        tick = time.monotonic()
        a7_query, a3_query = self.encoders.image(jpeg)
        a7_query = unit(a7_query)
        a3_full = unit(a3_query)
        a3_query = a3_full if self.projection is None else self.projection.apply(a3_query)
        timings["embed_photo"] = _ms(tick)
        tick = time.monotonic()
        a7 = np.where(mask, self.gallery.a7(a7_query), -np.inf).astype(np.float32)
        a3 = np.where(mask, self.gallery.a3(a3_query), -np.inf).astype(np.float32)
        if self.exact is not None:
            a3 = shortlist_cosines(self.exact, a3_full, a7, a3)
        timings["vectors_photo"] = _ms(tick)
        photo_type, _ = self.guess.photo_type(a7_query)
        chroma = chroma_spread(jpeg)
        bucket = self.routing.choose(photo_type, chroma)
        return PhotoCosines(a7=a7, a3=a3, bucket=bucket, photo_type=photo_type, chroma=chroma)

    def _photo(self, data: bytes, mask: NDArray[np.bool_], timings: dict[str, float]) -> _Photo:
        cosines = self._cosines(prepare_photo(data), mask, timings)
        a7, a3, bucket = cosines.a7, cosines.a3, cosines.bucket
        photo_type, chroma = cosines.photo_type, cosines.chroma
        w = bucket.a7_weight
        e7 = ranking.entity_max(a7, self.catalog)
        e3 = ranking.entity_max(a3, self.catalog)
        signals = [s for s in (ranking.Signal(e7, w), ranking.Signal(e3, 1 - w)) if s.weight > 0]
        return _Photo(
            signals=signals,
            per_image=weighted(a7, a3, w),
            e7=e7,
            e3=e3,
            a7_weight=w,
            cutoff=bucket.cutoff,
            bucket=bucket.name,
            photo_type=photo_type,
            chroma=chroma,
        )

    def _words(
        self,
        text: str,
        mask: NDArray[np.bool_],
        timings: dict[str, float],
        weights: TextWeights,
        key: str = "text",
    ) -> _Words:
        tick = time.monotonic()
        query = unit(self.encoders.text(text))
        timings[f"embed_{key}"] = _ms(tick)
        tick = time.monotonic()
        a7 = np.where(mask, self.gallery.a7(query), -np.inf).astype(np.float32)
        per_record = self.bm25.scores(text)
        signals = word_signals(self.catalog, a7, per_record, mask, weights)
        timings[f"vectors_{key}"] = _ms(tick)
        return _Words(signals=signals, per_image=a7, found=bool((per_record > 0).any()))

    # --- cards -----------------------------------------------------------------------------

    def _card(
        self,
        entity: int,
        per_image: Floats,
        mask: NDArray[np.bool_],
        score: float,
        level: Level,
    ) -> Card | None:
        catalog = self.catalog
        rows = catalog.entity_image_rows(entity)
        if ranking.best_image(rows, per_image, mask) is None:
            return None
        pop = rows[catalog.image_record[rows] >= 0]
        shown = ranking.best_image(pop, per_image, mask)
        if shown is None and pop.size:  # the filters leave only a Commons photo
            shown = min((int(r) for r in pop), key=lambda r: (not catalog.images[r].is_cover, r))
        image = None if shown is None else catalog.images[shown]
        record = (None if shown is None else catalog.record_of(shown)) or _first(
            catalog.entity_records(entity)
        )
        if record is None:
            return None
        matched: str | None = None
        if image is not None and not image.is_cover:
            matched = MATCHED_LABELS.get(image.image_type, "Autre vue")
        return Card(
            entity_id=catalog.entity_ids[entity],
            title=record.display_title,
            place=record.place,
            collection=record.collection,
            collection_label=COLLECTION_LABELS.get(record.collection, record.collection),
            pop_url=record.pop_url,
            image_url=None if image is None else display_url(image, CARD_SIDE),
            image_fallback_url=None if image is None else image.url,
            matched=matched,
            photo_count=int(pop.size),
            score=round(score, 4),
            level=level,
            level_label=LEVEL_LABELS[level],
        )


@dataclass(frozen=True, slots=True)
class PhotoCosines:
    a7: Floats
    """EmbeddingGemma 2 cosine of the photo with every image, in catalog order."""
    a3: Floats
    """MegaLoc, the same."""
    bucket: Bucket
    photo_type: PhotoType
    chroma: float


@dataclass(frozen=True, slots=True)
class _Photo:
    signals: list[ranking.Signal]
    per_image: Floats
    e7: Floats
    e3: Floats
    a7_weight: float
    cutoff: float
    bucket: str
    photo_type: PhotoType
    chroma: float

    def confidence(self, entity: int) -> float:
        """The routed mix of the entity's raw cosines, as T8 measured it (`notfound`)."""
        return float(
            weighted(self.e7[entity : entity + 1], self.e3[entity : entity + 1], self.a7_weight)[0]
        )


@dataclass(frozen=True, slots=True)
class _Archive:
    mount: bool = False
    query: bytes | None = None
    """The print alone, when the photo has a mount: what the photo search embeds."""
    box: tuple[float, float, float, float] | None = None
    caption: str = ""
    read: bool = False
    place: CaptionPlace | None = None


def _shares(picture: Picture) -> tuple[float, float, float, float]:
    w, h = picture.width, picture.height
    return (
        round(picture.left / w, 4),
        round(picture.top / h, 4),
        round(picture.right / w, 4),
        round(picture.bottom / h, 4),
    )


@dataclass(frozen=True, slots=True)
class _Words:
    signals: list[ranking.Signal]
    per_image: Floats
    found: bool
    """Some record holds one of the words (BM25 matched)."""


def word_signals(
    catalog: Catalog,
    a7: Floats,
    bm25_per_record: Floats,
    mask: NDArray[np.bool_],
    weights: TextWeights,
) -> list[ranking.Signal]:
    """The two word signals per entity, weighted: the A7 text vector's cosine with every
    image (`a7`, in catalog order) and BM25 over the record text, each image taking its
    record's score. Images `mask` removes score -inf. T7 scoring (`miroir.eval.wordsearch`)
    calls this too, so the measured ranking is the site's."""
    records = catalog.image_record
    bm25 = np.where(records >= 0, bm25_per_record[np.maximum(records, 0)], 0.0)
    bm25 = np.where(mask, bm25, -np.inf).astype(np.float32)
    signals = [
        ranking.Signal(ranking.entity_max(np.where(mask, a7, -np.inf), catalog), weights.a7),
        ranking.Signal(ranking.entity_max(bm25, catalog), weights.bm25),
    ]
    return [s for s in signals if s.weight > 0]


def prepare_photo(data: bytes) -> bytes:
    """A query photo as the gallery images were (`images.fetch.to_jpeg`): at most 512 px,
    LANCZOS, RGB JPEG at the same quality. Turned upright first, as a phone camera means it."""
    try:
        with PILImage.open(io.BytesIO(data)) as opened:
            upright = ImageOps.exif_transpose(opened)
            upright.thumbnail((LONG_SIDE, LONG_SIDE), PILImage.Resampling.LANCZOS)
            rgb = upright.convert("RGB")
    except (OSError, PILImage.DecompressionBombError) as err:  # UnidentifiedImageError too
        msg = f"cannot read the photo: {err}"
        raise ValueError(msg) from err
    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def record_text(record: CatalogRecord) -> str:
    """Every field worth searching by words, as `miroir.eval.bm25.record_text` joins them."""
    return " ".join(
        [
            record.title,
            record.description,
            *record.denomination,
            *record.domain,
            *record.authors,
            *record.period,
            record.commune or "",
            record.departement or "",
            record.region or "",
        ]
    )


def unit(vector: Floats) -> Floats:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm == 0:
        msg = "query vector is zero or not finite"
        raise ValueError(msg)
    return v / norm


def read_projection(directory: Path) -> Projection | None:
    path = directory / A3_PROJECTION_FILE
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return Projection(
            mean=np.asarray(data["mean"], dtype=np.float32),
            components=np.asarray(data["components"], dtype=np.float32),
        )


def load_engine(
    directory: Path,
    encoders: Encoders,
    gallery: GalleryVectors,
    captions: CaptionReader | None = None,
) -> Engine:
    """An engine over one index bundle; the caller brings the models and the vectors."""
    start = time.monotonic()
    catalog = read_catalog(directory / CATALOG_FILE)
    logger.info(
        "catalog: %d images, %d records, %d entities in %.1f s",
        len(catalog.images),
        len(catalog.records),
        catalog.entity_count,
        time.monotonic() - start,
    )
    return Engine(
        catalog=catalog,
        routing=read_routing(directory / ROUTING_FILE),
        guess=read_guess(directory / GUESS_FILE),
        encoders=encoders,
        gallery=gallery,
        projection=read_projection(directory),
        exact=read_exact(directory),
        places=read_places(directory / PLACES_FILE, catalog.entity_ids),
        sure=read_sure(directory / SURE_FILE),
        levels=read_levels(directory / LEVELS_FILE),
        captions=captions,
    )


def _first(records: list[CatalogRecord]) -> CatalogRecord | None:
    return records[0] if records else None


def _ms(since: float) -> float:
    return round(1000 * (time.monotonic() - since), 1)
