"""One search: a photo and/or words, plus filters, in; cards and a "not found" flag out.

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

from miroir.eval.bm25 import Bm25
from miroir.images.colour import chroma_spread
from miroir.images.fetch import JPEG_QUALITY, LONG_SIDE
from miroir.web import ranking
from miroir.web.catalog import (
    CARD_SIDE,
    CATALOG_FILE,
    COLLECTION_LABELS,
    MATCHED_LABELS,
    Catalog,
    CatalogRecord,
    Filters,
    Source,
    display_url,
    read_catalog,
)
from miroir.web.nearby import NEAR_KM, PLACES_FILE, EntityPlaces, Position, read_places
from miroir.web.routing import (
    GUESS_FILE,
    ROUTING_FILE,
    Guess,
    PhotoType,
    Routing,
    read_guess,
    read_routing,
)

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]

#: Cards per page of results.
RESULTS = 24
#: Longest text query kept, in characters.
MAX_TEXT = 300
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


@dataclass(frozen=True, slots=True)
class Card:
    """One building or object (functional spec §6)."""

    entity_id: str
    title: str
    place: str
    collection: str
    collection_label: str
    pop_url: str
    image_url: str
    """The image that best matches the query."""
    image_fallback_url: str
    """The same file straight from the bucket, if the first URL fails."""
    matched: str | None
    """How the matched image differs from the cover ("Détail", "Revers"...), or None."""
    credit: str | None
    """Author and licence when the matched image is a Commons photo."""
    photo_count: int
    score: float


@dataclass(frozen=True, slots=True)
class SearchResponse:
    mode: str
    """"photo", "text" or "photo+text"."""
    match: bool | None
    """Photo searches: False when the top result is below the cut-off. None for words only."""
    confidence: float | None
    guessed_type: str | None
    bucket: str | None
    chroma: float | None
    """The photo's chroma spread (`images.colour`): near 0 for black and white."""
    cards: tuple[Card, ...]
    timings_ms: dict[str, float] = field(default_factory=dict[str, float])
    nearby: int | None = None
    """With a position: entities known within `near_km` of it (the near ones rank higher)."""
    near_km: float | None = None


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
        places: EntityPlaces | None = None,
    ) -> None:
        self.catalog = catalog
        self.places = places
        self.routing = routing
        self.guess = guess
        self.encoders = encoders
        self.gallery = gallery
        self.projection = projection
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
        signals: list[ranking.Signal] = []
        per_image: Floats | None = None
        photo: _Photo | None = None
        if request.image is not None:
            photo = self._photo(request.image, mask, timings)
            signals += photo.signals
            per_image = photo.per_image
        if text:
            words = self._words(text, mask, timings)
            scale = 1.0 if photo is None else self.routing.text.refine
            signals += [ranking.Signal(s.per_entity, s.weight * scale) for s in words.signals]
            if per_image is None:
                per_image = words.per_image
        assert per_image is not None
        tick = time.monotonic()
        fused = ranking.fuse(signals)
        nearby = None
        if request.position is not None and self.places is not None:
            fused = fused + self.places.prior(request.position)  # -inf (filtered out) stays
            nearby = int(self.places.near(request.position).sum())
        top = ranking.top_entities(fused, max(1, min(request.limit, 100)))
        cards = tuple(
            card
            for row in top
            if (card := self._card(int(row), per_image, mask, float(fused[row]))) is not None
        )
        confidence = match = None
        if photo is not None and len(top):
            confidence = photo.confidence(int(top[0]))
            match = confidence >= photo.cutoff
        timings["rank"] = _ms(tick)
        timings["total"] = _ms(start)
        return SearchResponse(
            mode="photo+text" if photo and text else ("photo" if photo else "text"),
            match=match,
            confidence=confidence,
            guessed_type=None if photo is None else photo.photo_type.value,
            bucket=None if photo is None else photo.bucket,
            chroma=None if photo is None else photo.chroma,
            cards=cards,
            timings_ms=timings,
            nearby=nearby,
            near_km=None if nearby is None else NEAR_KM,
        )

    # --- the two kinds of query ------------------------------------------------------------

    def _photo(self, data: bytes, mask: NDArray[np.bool_], timings: dict[str, float]) -> _Photo:
        tick = time.monotonic()
        jpeg = prepare_photo(data)
        a7_query, a3_query = self.encoders.image(jpeg)
        a7_query = unit(a7_query)
        a3_query = unit(a3_query) if self.projection is None else self.projection.apply(a3_query)
        timings["embed_photo"] = _ms(tick)
        tick = time.monotonic()
        a7 = np.where(mask, self.gallery.a7(a7_query), -np.inf).astype(np.float32)
        a3 = np.where(mask, self.gallery.a3(a3_query), -np.inf).astype(np.float32)
        timings["vectors_photo"] = _ms(tick)
        photo_type, _ = self.guess.photo_type(a7_query)
        chroma = chroma_spread(jpeg)
        bucket = self.routing.choose(photo_type, chroma)
        w = bucket.a7_weight
        e7 = ranking.entity_max(a7, self.catalog)
        e3 = ranking.entity_max(a3, self.catalog)
        signals = [s for s in (ranking.Signal(e7, w), ranking.Signal(e3, 1 - w)) if s.weight > 0]
        return _Photo(
            signals=signals,
            per_image=(w * a7 + (1 - w) * a3).astype(np.float32),
            e7=e7,
            e3=e3,
            a7_weight=w,
            cutoff=bucket.cutoff,
            bucket=bucket.name,
            photo_type=photo_type,
            chroma=chroma,
        )

    def _words(self, text: str, mask: NDArray[np.bool_], timings: dict[str, float]) -> _Words:
        tick = time.monotonic()
        query = unit(self.encoders.text(text))
        timings["embed_text"] = _ms(tick)
        tick = time.monotonic()
        a7 = np.where(mask, self.gallery.a7(query), -np.inf).astype(np.float32)
        per_record = self.bm25.scores(text)
        records = self.catalog.image_record
        bm25 = np.where(records >= 0, per_record[np.maximum(records, 0)], 0.0)
        bm25 = np.where(mask, bm25, -np.inf).astype(np.float32)
        weights = self.routing.text
        signals = [
            ranking.Signal(ranking.entity_max(a7, self.catalog), weights.a7),
            ranking.Signal(ranking.entity_max(bm25, self.catalog), weights.bm25),
        ]
        timings["vectors_text"] = _ms(tick)
        return _Words(signals=[s for s in signals if s.weight > 0], per_image=a7)

    # --- cards -----------------------------------------------------------------------------

    def _card(
        self, entity: int, per_image: Floats, mask: NDArray[np.bool_], score: float
    ) -> Card | None:
        catalog = self.catalog
        rows = catalog.entity_image_rows(entity)
        best = ranking.best_image(rows, per_image, mask)
        if best is None:
            return None
        image = catalog.images[best]
        record = catalog.record_of(best) or _first(catalog.entity_records(entity))
        if record is None:
            return None
        matched: str | None = None
        credit: str | None = None
        if image.source is Source.COMMONS:
            matched = "Photo Wikimedia Commons"
            credit = ", ".join(p for p in (image.author, image.licence) if p) or None
        elif not image.is_cover:
            matched = MATCHED_LABELS.get(image.image_type, "Autre vue")
        return Card(
            entity_id=catalog.entity_ids[entity],
            title=record.display_title,
            place=record.place,
            collection=record.collection,
            collection_label=COLLECTION_LABELS.get(record.collection, record.collection),
            pop_url=record.pop_url,
            image_url=display_url(image, CARD_SIDE),
            image_fallback_url=image.url,
            matched=matched,
            credit=credit,
            photo_count=len(rows),
            score=round(score, 4),
        )


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
        w = self.a7_weight
        return float(w * self.e7[entity] + (1 - w) * self.e3[entity])


@dataclass(frozen=True, slots=True)
class _Words:
    signals: list[ranking.Signal]
    per_image: Floats


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


def load_engine(directory: Path, encoders: Encoders, gallery: GalleryVectors) -> Engine:
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
        places=read_places(directory / PLACES_FILE, catalog.entity_ids),
    )


def _first(records: list[CatalogRecord]) -> CatalogRecord | None:
    return records[0] if records else None


def _ms(since: float) -> float:
    return round(1000 * (time.monotonic() - since), 1)
