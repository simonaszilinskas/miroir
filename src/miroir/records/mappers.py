"""Turn one raw POP export line into a `Record` (technical spec §3.2).

Each base is a `FieldMap`: which POP field codes feed which record field. A new base
(Mérimée, Palissy, ...) needs only a new `FieldMap` entry in `FIELD_MAPS`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from miroir.config import POP_NOTICE_URL
from miroir.records.images import (
    ImageSkip,
    asset_url,
    image_type_from_path,
    make_image_id,
    skip_reason,
)
from miroir.records.model import Collection, Image, ImageType, Record

RawRecord = Mapping[str, object]

DESCRIPTION_MAX_CHARS = 2000


class RecordSkip(StrEnum):
    NO_IMAGE = "no_image"
    """No usable image is left once image skips apply."""


@dataclass(frozen=True, slots=True)
class MapResult:
    record: Record | None
    record_skip: RecordSkip | None
    image_skips: tuple[ImageSkip, ...]


def _ref(raw: RawRecord) -> str:
    return _text(raw, "REF")


@dataclass(frozen=True, slots=True)
class FieldMap:
    """POP field codes for each record field. Fields are read in order; empty ones are ignored."""

    title: tuple[str, ...]
    """First non-empty value wins."""
    description: tuple[str, ...]
    """Non-empty values are joined, then cut to `DESCRIPTION_MAX_CHARS`."""
    denomination: tuple[str, ...]
    domain: tuple[str, ...]
    authors: tuple[str, ...]
    period: tuple[str, ...]
    region: tuple[str, ...]
    departement: tuple[str, ...]
    commune: tuple[str, ...]
    images: str
    """Field holding the image paths, relative to the POP asset host."""
    publishable: Callable[[RawRecord], bool]
    entity_id: Callable[[RawRecord], str] = field(default=_ref)
    decade_of: tuple[str, ...] = ()
    """Date fields whose decade is added to `period` ("années 1940")."""
    linked: tuple[str, ...] = ()
    """Fields listing refs of related records in the same base."""
    photographers: tuple[str, ...] = ()
    photo_date: tuple[str, ...] = ()
    """First non-empty value wins, kept as written."""


def _memoire_entity_id(raw: RawRecord) -> str:
    """First Palissy object id (PM) in `LBASE`, else first Mérimée building id (PA), else REF.

    A photo linked to both shows the object, so the object wins.
    """
    linked = _values(raw, "LBASE")
    for prefix in ("PM", "PA"):
        found = next((value for value in linked if value.startswith(prefix)), None)
        if found is not None:
            return found
    return _ref(raw)


def _joconde_publishable(raw: RawRecord) -> bool:
    return _text(raw, "DIFFU").lower() != "non"


def _memoire_publishable(raw: RawRecord) -> bool:
    # DIFF values seen: "", "reproduction soumise à autorisation du titulaire ...",
    # "communication libre, reproduction soumise à autorisation".
    return "autorisation" not in _text(raw, "DIFF").lower()


FIELD_MAPS: dict[Collection, FieldMap] = {
    Collection.JOCONDE: FieldMap(
        title=("TITR",),
        description=("DESC", "HIST"),
        denomination=("DENO",),
        domain=("DOMN",),
        authors=("AUTR",),
        period=("EPOQ", "PERI", "MILL"),
        region=("REGION",),
        departement=("DPT",),
        commune=("VILLE_M",),
        images="IMG",
        publishable=_joconde_publishable,
    ),
    Collection.MEMOIRE: FieldMap(
        title=("EDIF", "LEG"),
        description=("LEG",),
        denomination=("DENO", "MCL"),
        domain=("DOM",),
        authors=("AUTR", "AUTP"),
        period=(),
        decade_of=("DATPV",),
        region=("REG",),
        departement=("DPT_LETTRE",),
        commune=("COM",),
        # IMG is the largest variant; VIDEO lists the same file in other sizes.
        images="IMG",
        publishable=_memoire_publishable,
        entity_id=_memoire_entity_id,
        photographers=("AUTP",),
        photo_date=("DATPV",),
    ),
    Collection.ROSE_VALLAND: FieldMap(
        title=("TITR",),
        description=("DESC", "HIST"),
        denomination=("DENO",),
        domain=("DOMN",),
        authors=("AUTR",),
        period=("MILL",),
        region=("REG",),
        departement=("DPT",),
        commune=("COM",),
        images="VIDEO",
        publishable=lambda _raw: True,
        # Jeu de Paume: a room photo lists the artworks it shows, and each artwork its room(s).
        linked=("RENV",),
    ),
}


def map_record(collection: Collection, raw: RawRecord) -> MapResult:
    fmap = FIELD_MAPS[collection]
    ref = _ref(raw)
    if not ref:
        msg = f"{collection} record without REF"
        raise ValueError(msg)
    entity_id = fmap.entity_id(raw)

    paths = list(_values(raw, fmap.images))
    if not paths:
        # The export then puts a 1x1 GIF data URL in imageUrl: count it as a placeholder.
        paths = [_text(raw, "imageUrl")]
    kept, image_skips = _select_images(paths)
    if not kept:
        return MapResult(record=None, record_skip=RecordSkip.NO_IMAGE, image_skips=image_skips)

    images = _build_images(kept, ref=ref, entity_id=entity_id, collection=collection)
    period = (*_keywords(raw, fmap.period), *_decades(raw, fmap.decade_of))
    record = Record(
        ref=ref,
        collection=collection,
        entity_id=entity_id,
        title=next(iter(_texts(raw, fmap.title)), ""),
        description="\n\n".join(_texts(raw, fmap.description))[:DESCRIPTION_MAX_CHARS],
        denomination=_unique(_keywords(raw, fmap.denomination)),
        domain=_unique(_keywords(raw, fmap.domain)),
        authors=_unique(_keywords(raw, fmap.authors)),
        period=_unique(period),
        region=_first(raw, fmap.region),
        departement=_first(raw, fmap.departement),
        commune=_first(raw, fmap.commune),
        publishable=fmap.publishable(raw),
        pop_url=f"{POP_NOTICE_URL}/{collection.mongo_collection}/{ref}",
        linked_refs=_unique(_keywords(raw, fmap.linked)),
        images=images,
        photographers=_unique(_keywords(raw, fmap.photographers)),
        photo_date=next(iter(_texts(raw, fmap.photo_date)), ""),
    )
    return MapResult(record=record, record_skip=None, image_skips=image_skips)


def _select_images(paths: Iterable[str]) -> tuple[list[str], tuple[ImageSkip, ...]]:
    kept: list[str] = []
    skips: list[ImageSkip] = []
    for path in paths:
        reason = skip_reason(path)
        if reason is None and path in kept:
            reason = ImageSkip.DUPLICATE
        if reason is None:
            kept.append(path)
        else:
            skips.append(reason)
    return kept, tuple(skips)


def _build_images(
    paths: list[str], *, ref: str, entity_id: str, collection: Collection
) -> tuple[Image, ...]:
    types = [image_type_from_path(path) for path in paths]
    cover = types.index(ImageType.MAIN) if ImageType.MAIN in types else 0
    images: list[Image] = []
    for position, (path, image_type) in enumerate(zip(paths, types, strict=True)):
        url = asset_url(path)
        images.append(
            Image(
                image_id=make_image_id(ref, url),
                ref=ref,
                entity_id=entity_id,
                collection=collection,
                url=url,
                image_type=image_type,
                position=position,
                is_cover=position == cover,
            )
        )
    return tuple(images)


# --- raw field access -------------------------------------------------------

_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")


def _values(raw: RawRecord, code: str) -> tuple[str, ...]:
    """A field as a tuple of non-empty strings. Strings holding "a ; b" are split."""
    value = raw.get(code)
    items: list[object]
    if isinstance(value, list):
        items = list(value)  # pyright: ignore[reportUnknownArgumentType]
    elif value is None:
        items = []
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        out.extend(part.strip() for part in item.split(";") if part.strip())
    return tuple(out)


def _text(raw: RawRecord, code: str) -> str:
    value = raw.get(code)
    return value.strip() if isinstance(value, str) else ""


def _keywords(raw: RawRecord, codes: Iterable[str]) -> tuple[str, ...]:
    """Every non-empty value of the codes, in order, split on ";"."""
    return tuple(value for code in codes for value in _values(raw, code))


def _texts(raw: RawRecord, codes: Iterable[str]) -> tuple[str, ...]:
    """Every non-empty free-text value of the codes, in order, kept whole."""
    return tuple(text for code in codes if (text := _text(raw, code)))


def _first(raw: RawRecord, codes: Iterable[str]) -> str | None:
    return next(iter(_keywords(raw, codes)), None)


def _decades(raw: RawRecord, codes: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in _texts(raw, codes):
        match = _YEAR_RE.search(value)
        if match:
            out.append(f"années {int(match.group(1)) // 10 * 10}")
    return tuple(out)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
