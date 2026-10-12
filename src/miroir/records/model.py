"""The common record model (technical spec §3.2 and §3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Collection(StrEnum):
    """A POP base. The value is the export's `base` field."""

    JOCONDE = "joconde"
    MEMOIRE = "memoire"
    ROSE_VALLAND = "rose-valland"

    @property
    def mongo_collection(self) -> str:
        """Name POP uses in its URLs."""
        return _MONGO[self]


_MONGO = {
    Collection.JOCONDE: "joconde",
    Collection.MEMOIRE: "memoire",
    Collection.ROSE_VALLAND: "mnr",
}


class ImageType(StrEnum):
    MAIN = "main"
    DETAIL = "detail"
    BACK = "back"
    INSCRIPTION = "inscription"
    LABEL = "label"


@dataclass(frozen=True, slots=True)
class Image:
    image_id: str
    ref: str
    entity_id: str
    collection: Collection
    url: str
    image_type: ImageType
    position: int
    """0-based rank of the image within its record, after skips."""
    is_cover: bool
    """The record's first `main` image (or its first image when none is `main`)."""


@dataclass(frozen=True, slots=True)
class Record:
    ref: str
    collection: Collection
    entity_id: str
    """Building or object the record shows. Mémoire: first PM `LBASE` id, else PA, else `ref`."""
    title: str
    description: str
    denomination: tuple[str, ...]
    domain: tuple[str, ...]
    authors: tuple[str, ...]
    period: tuple[str, ...]
    region: str | None
    departement: str | None
    commune: str | None
    publishable: bool
    """False when the rights holder must approve reproduction. Gate for outside model services."""
    pop_url: str
    linked_refs: tuple[str, ...]
    """Refs of related records in the same base (Rose Valland `RENV`): room photo <-> artwork."""
    images: tuple[Image, ...]
    photographers: tuple[str, ...] = ()
    """Mémoire `AUTP`: who took the photo. Empty elsewhere."""
    photo_date: str = ""
    """Mémoire `DATPV` as written, mostly a year. With `photographers` it marks one photo shoot."""

    @property
    def cover(self) -> Image:
        return next(image for image in self.images if image.is_cover)
