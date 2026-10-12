"""Wikimedia Commons photos used as test queries (technical spec §7: T2, T2b, T8).

They are queries only: no Commons photo is ever added to a gallery. Every one is openly
licensed, so it may go to any model service, closed ones included.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

#: Every Commons image id starts with this, so it can never clash with a POP image id
#: (16 hex characters) and code can tell the two apart.
COMMONS_ID_PREFIX = "commons-"


class CommonsSet(StrEnum):
    """The test set a Commons photo serves."""

    T2 = "T2"
    """Phone building: a photo of a Mérimée building (PA) that has photos in Mémoire."""
    T2B = "T2b"
    """Museum work: a photo of a Joconde work that has an image in our table."""
    T8 = "T8"
    """Not found: a French monument or work with no image in our table."""


class Source(StrEnum):
    """Where on Commons the photo was found."""

    P18 = "p18"
    """The Wikidata item's image (P18)."""
    CATEGORY = "category"
    """A file in the item's Commons category (P373)."""
    SUBCATEGORY = "subcategory"
    """A file in a subcategory of it about the interior or details."""


class View(StrEnum):
    """What part of a building the photo shows, guessed from its title and categories."""

    EXTERIOR = "exterior"
    INTERIOR = "interior"
    DETAIL = "detail"
    ARTWORK = "artwork"
    """A museum work: no view is guessed."""


@dataclass(frozen=True, slots=True)
class CommonsPhoto:
    """One Commons file chosen as a query, with what its licence asks us to keep."""

    image_id: str
    set_id: CommonsSet
    entity_id: str
    """The answer: a Mérimée id (PA...) or a Joconde ref, from the Wikidata item."""
    wikidata_id: str
    file_title: str
    """"File:..." on Commons."""
    page_url: str
    url: str
    """The original file."""
    thumb_url: str
    """A copy 960 px wide (the original when it is smaller)."""
    width: int
    height: int
    licence: str
    licence_url: str
    author: str
    """Plain text, as the file page credits it."""
    credit: str
    taken: str
    """When the photo was taken (EXIF DateTimeOriginal, else the file page's date), as given."""
    latitude: float | None
    longitude: float | None
    """Camera position from EXIF GPS, else from the file page's location."""
    camera: str
    """EXIF make and model."""
    source: Source
    category: str
    """The Commons category it was found in; empty for P18."""
    view: View
    phash_distance: int | None = None
    """T2b only: bits between its perceptual hash and the nearest image of the Joconde record."""


def commons_image_id(file_title: str) -> str:
    """Stable id from the file title: `commons-` and 16 hex characters."""
    digest = hashlib.sha1(file_title.encode(), usedforsecurity=False).hexdigest()[:16]
    return f"{COMMONS_ID_PREFIX}{digest}"


def is_commons_image(image_id: str) -> bool:
    return image_id.startswith(COMMONS_ID_PREFIX)
