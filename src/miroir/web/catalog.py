"""What the website knows about the indexed images and records: cards, filters, record pages.

A catalog is read from an index bundle (`catalog.json.gz`, written by `miroir.web.bundle`).
Row `i` of `images` is row `i` of the bundle's vector matrices. Images group into entities
(functional spec §6): a building or an object, with all its photos.

An image may come from POP (it belongs to a record) or be an extra photo of an entity from
Wikimedia Commons (`source` "commons", no record, licence and author kept for the card).
"""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter

from miroir.config import POP_ASSET_HOST, POP_IIIF_IMAGE_HOST
from miroir.web import facets

CATALOG_FILE = "catalog.json.gz"
#: Longest side of a card image and of a record page image, served by the IIIF service when
#: the bucket file is large (Rose Valland files reach 20 MB).
CARD_SIDE = 640
PAGE_SIDE = 1400
#: Images shown on a record page, at most.
PAGE_IMAGES = 24

Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]

COLLECTION_LABELS = {
    "joconde": "Joconde",
    "memoire": "Mémoire",
    "rose-valland": "Rose Valland",
}
COLLECTION_HINTS = {
    "joconde": "Collections des musées de France",
    "memoire": "Photographies du patrimoine",
    "rose-valland": "Œuvres récupérées après 1945 (MNR)",
}
#: How a card names the matched image when it is not the record's cover.
MATCHED_LABELS = {
    "main": "Autre vue",
    "detail": "Détail",
    "back": "Revers",
    "inscription": "Inscription",
    "label": "Étiquette ou cachet",
}


class Source(StrEnum):
    POP = "pop"
    COMMONS = "commons"


@dataclass(frozen=True, slots=True)
class CatalogImage:
    image_id: str
    entity_id: str
    ref: str | None
    """Its record; None for a Commons photo."""
    url: str
    image_type: str
    is_cover: bool
    source: Source = Source.POP
    licence: str = ""
    author: str = ""
    page_url: str = ""
    """The Commons file page, for the credit line."""


@dataclass(frozen=True, slots=True)
class CatalogRecord:
    ref: str
    collection: str
    entity_id: str
    title: str
    description: str
    denomination: tuple[str, ...]
    domain: tuple[str, ...]
    authors: tuple[str, ...]
    period: tuple[str, ...]
    region: str | None
    departement: str | None
    commune: str | None
    pop_url: str

    @property
    def display_title(self) -> str:
        if self.title.strip():
            return self.title.strip()
        if self.denomination:
            return self.denomination[0][:1].upper() + self.denomination[0][1:]
        return "Sans titre"

    @property
    def place(self) -> str:
        parts = [p for p in (self.commune, self.departement) if p]
        if len(parts) == 2 and parts[0] == parts[1]:
            parts = parts[:1]
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class CatalogData:
    """The file's content: images in vector-row order, then records."""

    images: tuple[CatalogImage, ...]
    records: tuple[CatalogRecord, ...]


_DATA = TypeAdapter(CatalogData)


def write_catalog(path: Path, data: CatalogData) -> None:
    path.write_bytes(gzip.compress(_DATA.dump_json(data), compresslevel=6))


def read_catalog(path: Path) -> Catalog:
    return Catalog(_DATA.validate_json(gzip.decompress(path.read_bytes())))


@dataclass(frozen=True, slots=True)
class Filters:
    """F3: narrow a search before ranking. Empty means no filter."""

    collections: tuple[str, ...] = ()
    region: str | None = None
    departement: str | None = None
    domain: str | None = None
    period: str | None = None

    @property
    def empty(self) -> bool:
        return not (self.collections or self.region or self.departement or self.domain) and (
            self.period is None
        )


@dataclass(frozen=True, slots=True)
class FacetValue:
    value: str
    count: int


@dataclass(frozen=True, slots=True)
class FacetOptions:
    """Menu values for each filter, with how many records hold them."""

    collections: tuple[FacetValue, ...]
    regions: tuple[FacetValue, ...]
    departements: dict[str, tuple[FacetValue, ...]]
    """Per region; key "" holds every département."""
    domains: tuple[FacetValue, ...]
    periods: tuple[FacetValue, ...]


#: Domains offered in the menu, most common first.
TOP_DOMAINS = 40
#: A region with fewer records is left out of the menu (foreign places, typos).
MIN_REGION_RECORDS = 5


class Catalog:
    """Images, records and entities, with the index arrays search needs."""

    def __init__(self, data: CatalogData) -> None:
        self.images = data.images
        self.records = data.records
        record_row = {r.ref: i for i, r in enumerate(self.records)}
        if len(record_row) != len(self.records):
            msg = "catalog holds duplicate record refs"
            raise ValueError(msg)
        self.entity_ids: tuple[str, ...] = tuple(
            sorted({i.entity_id for i in self.images} | {r.entity_id for r in self.records})
        )
        self.entity_row = {e: i for i, e in enumerate(self.entity_ids)}
        self.image_entity: Ints = np.fromiter(
            (self.entity_row[i.entity_id] for i in self.images), np.int64, len(self.images)
        )
        missing = [i.ref for i in self.images if i.ref is not None and i.ref not in record_row]
        if missing:
            msg = f"{len(missing)} image(s) point to no record, e.g. {missing[:3]}"
            raise ValueError(msg)
        self.image_record: Ints = np.fromiter(
            (-1 if i.ref is None else record_row[i.ref] for i in self.images),
            np.int64,
            len(self.images),
        )
        self.record_entity: Ints = np.fromiter(
            (self.entity_row[r.entity_id] for r in self.records), np.int64, len(self.records)
        )
        # Images sorted by entity, for a max per entity with `np.maximum.reduceat`.
        self.entity_order: Ints = np.argsort(self.image_entity, kind="stable")
        sorted_entities = self.image_entity[self.entity_order]
        self.entity_starts: Ints = np.flatnonzero(np.r_[True, np.diff(sorted_entities) != 0])
        self.entities_with_images: Ints = sorted_entities[self.entity_starts]
        self._regions = tuple(facets.region(r.region) for r in self.records)
        self._domains = tuple(frozenset(facets.domain(d) for d in r.domain) for r in self.records)
        self._periods = tuple(frozenset(facets.periods(r.period)) for r in self.records)
        self.image_row = {i.image_id: n for n, i in enumerate(self.images)}

    @property
    def entity_count(self) -> int:
        return len(self.entity_ids)

    # --- filters ---------------------------------------------------------------------------

    def record_mask(self, filters: Filters) -> Bools:
        keep = np.ones(len(self.records), dtype=bool)
        if filters.collections:
            wanted = set(filters.collections)
            keep &= np.fromiter((r.collection in wanted for r in self.records), bool)
        if filters.region:
            keep &= np.fromiter((r == filters.region for r in self._regions), bool)
        if filters.departement:
            keep &= np.fromiter((r.departement == filters.departement for r in self.records), bool)
        if filters.domain:
            keep &= np.fromiter((filters.domain in d for d in self._domains), bool)
        if filters.period:
            keep &= np.fromiter((filters.period in p for p in self._periods), bool)
        return keep

    def image_mask(self, filters: Filters) -> Bools:
        """Images a filtered search may return. A Commons photo follows its entity's records."""
        if filters.empty:
            return np.ones(len(self.images), dtype=bool)
        records = self.record_mask(filters)
        entities = np.zeros(self.entity_count, dtype=bool)
        entities[self.record_entity[records]] = True
        from_pop = self.image_record >= 0
        out = np.empty(len(self.images), dtype=bool)
        out[from_pop] = records[self.image_record[from_pop]]
        out[~from_pop] = entities[self.image_entity[~from_pop]]
        return out

    def facet_options(self) -> FacetOptions:
        def counted(values: Sequence[str]) -> dict[str, int]:
            out: dict[str, int] = {}
            for v in values:
                out[v] = out.get(v, 0) + 1
            return out

        def sorted_by_name(counts: dict[str, int]) -> tuple[FacetValue, ...]:
            return tuple(FacetValue(v, n) for v, n in sorted(counts.items(), key=_name_key))

        collections = counted([r.collection for r in self.records])
        regions = {
            k: n
            for k, n in counted([r for r in self._regions if r]).items()
            if n >= MIN_REGION_RECORDS
        }
        by_region: dict[str, dict[str, int]] = {"": {}}
        for record, reg in zip(self.records, self._regions, strict=True):
            if not record.departement:
                continue
            for key in ("", reg or ""):
                bucket = by_region.setdefault(key, {})
                bucket[record.departement] = bucket.get(record.departement, 0) + 1
        domains = counted([d for ds in self._domains for d in ds])
        top_domains = dict(sorted(domains.items(), key=lambda kv: -kv[1])[:TOP_DOMAINS])
        periods = counted([p for ps in self._periods for p in ps])
        return FacetOptions(
            collections=tuple(
                FacetValue(c, collections[c]) for c in COLLECTION_LABELS if c in collections
            ),
            regions=sorted_by_name(regions),
            departements={
                k: sorted_by_name(v) for k, v in by_region.items() if k in regions or not k
            },
            domains=sorted_by_name(top_domains),
            periods=tuple(FacetValue(p, periods[p]) for p in facets.PERIODS if p in periods),
        )

    # --- entities --------------------------------------------------------------------------

    def entity_image_rows(self, entity: int) -> Ints:
        """Image rows of one entity (by entity row), in catalog order."""
        where = int(np.searchsorted(self.entities_with_images, entity))
        if where >= len(self.entities_with_images) or self.entities_with_images[where] != entity:
            return np.zeros(0, dtype=np.int64)
        start = int(self.entity_starts[where])
        end = (
            int(self.entity_starts[where + 1])
            if where + 1 < len(self.entity_starts)
            else len(self.entity_order)
        )
        return np.sort(self.entity_order[start:end])

    def entity_records(self, entity: int) -> list[CatalogRecord]:
        return [self.records[int(i)] for i in np.flatnonzero(self.record_entity == entity)]

    def record_of(self, image_row: int) -> CatalogRecord | None:
        row = int(self.image_record[image_row])
        return None if row < 0 else self.records[row]


def _name_key(item: tuple[str, int]) -> str:
    return facets.fold(item[0])


def display_url(image: CatalogImage, side: int) -> str:
    """URL a browser loads for an image: the file itself, or a IIIF copy for big POP files.

    Rose Valland files are full-size scans; the IIIF service serves them scaled. Joconde and
    Mémoire files are already small. Commons photos come with their own thumbnail URL.
    """
    prefix = f"{POP_ASSET_HOST}/"
    if image.source is Source.POP and "/mnr/" in image.url and image.url.startswith(prefix):
        path = unquote(image.url.removeprefix(prefix))
        return f"{POP_IIIF_IMAGE_HOST}/{quote(path, safe='')}/full/!{side},{side}/0/default.jpg"
    return image.url
