"""Read and write `records.parquet` and `images.parquet`.

Downstream code loads records with `read_records()` / `read_images()` rather than
opening the Parquet files itself, so the schema lives in one place.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from miroir.commons.model import CommonsPhoto, CommonsSet, Source, View
from miroir.records.model import Collection, Image, ImageType, Record

_STRINGS = pa.list_(pa.string())

RECORDS_SCHEMA = pa.schema(
    [
        pa.field("ref", pa.string(), nullable=False),
        pa.field("collection", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("denomination", _STRINGS, nullable=False),
        pa.field("domain", _STRINGS, nullable=False),
        pa.field("authors", _STRINGS, nullable=False),
        pa.field("period", _STRINGS, nullable=False),
        pa.field("region", pa.string()),
        pa.field("departement", pa.string()),
        pa.field("commune", pa.string()),
        pa.field("publishable", pa.bool_(), nullable=False),
        pa.field("pop_url", pa.string(), nullable=False),
        pa.field("linked_refs", _STRINGS, nullable=False),
        pa.field("image_count", pa.int32(), nullable=False),
        pa.field("cover_image_id", pa.string(), nullable=False),
        pa.field("photographers", _STRINGS, nullable=False),
        pa.field("photo_date", pa.string(), nullable=False),
    ]
)

IMAGES_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("ref", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("collection", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("image_type", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("is_cover", pa.bool_(), nullable=False),
    ]
)


COMMONS_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("set_id", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("wikidata_id", pa.string(), nullable=False),
        pa.field("file_title", pa.string(), nullable=False),
        pa.field("page_url", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("thumb_url", pa.string(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        pa.field("licence", pa.string(), nullable=False),
        pa.field("licence_url", pa.string(), nullable=False),
        pa.field("author", pa.string(), nullable=False),
        pa.field("credit", pa.string(), nullable=False),
        pa.field("taken", pa.string(), nullable=False),
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("camera", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("view", pa.string(), nullable=False),
        pa.field("phash_distance", pa.int32()),
    ]
)


def write_records(records: Sequence[Record], records_path: Path, images_path: Path) -> None:
    """Write both files. Rows keep the order given."""
    records_path.parent.mkdir(parents=True, exist_ok=True)
    images_path.parent.mkdir(parents=True, exist_ok=True)
    record_rows = [
        {
            "ref": r.ref,
            "collection": r.collection.value,
            "entity_id": r.entity_id,
            "title": r.title,
            "description": r.description,
            "denomination": list(r.denomination),
            "domain": list(r.domain),
            "authors": list(r.authors),
            "period": list(r.period),
            "region": r.region,
            "departement": r.departement,
            "commune": r.commune,
            "publishable": r.publishable,
            "pop_url": r.pop_url,
            "linked_refs": list(r.linked_refs),
            "image_count": len(r.images),
            "cover_image_id": r.cover.image_id,
            "photographers": list(r.photographers),
            "photo_date": r.photo_date,
        }
        for r in records
    ]
    image_rows = [
        {
            "image_id": i.image_id,
            "ref": i.ref,
            "entity_id": i.entity_id,
            "collection": i.collection.value,
            "url": i.url,
            "image_type": i.image_type.value,
            "position": i.position,
            "is_cover": i.is_cover,
        }
        for r in records
        for i in r.images
    ]
    _write(pa.Table.from_pylist(record_rows, schema=RECORDS_SCHEMA), records_path)
    _write(pa.Table.from_pylist(image_rows, schema=IMAGES_SCHEMA), images_path)


def read_images(images_path: Path) -> list[Image]:
    rows = _read(images_path, IMAGES_SCHEMA).to_pylist()
    return [
        Image(
            image_id=str(row["image_id"]),
            ref=str(row["ref"]),
            entity_id=str(row["entity_id"]),
            collection=Collection(row["collection"]),
            url=str(row["url"]),
            image_type=ImageType(row["image_type"]),
            position=int(row["position"]),
            is_cover=bool(row["is_cover"]),
        )
        for row in rows
    ]


def read_records(records_path: Path, images_path: Path) -> list[Record]:
    """Load every record with its images, in file order."""
    by_ref: defaultdict[str, list[Image]] = defaultdict(list)
    for image in read_images(images_path):
        by_ref[image.ref].append(image)
    rows = _read(records_path, RECORDS_SCHEMA).to_pylist()
    return [
        Record(
            ref=str(row["ref"]),
            collection=Collection(row["collection"]),
            entity_id=str(row["entity_id"]),
            title=str(row["title"]),
            description=str(row["description"]),
            denomination=_strings(row["denomination"]),
            domain=_strings(row["domain"]),
            authors=_strings(row["authors"]),
            period=_strings(row["period"]),
            region=_optional(row["region"]),
            departement=_optional(row["departement"]),
            commune=_optional(row["commune"]),
            publishable=bool(row["publishable"]),
            pop_url=str(row["pop_url"]),
            linked_refs=_strings(row["linked_refs"]),
            images=tuple(sorted(by_ref[str(row["ref"])], key=lambda i: i.position)),
            photographers=_strings(row["photographers"]),
            photo_date=str(row["photo_date"]),
        )
        for row in rows
    ]


def write_commons(photos: Sequence[CommonsPhoto], path: Path) -> None:
    """Write the Commons query photos (technical spec §7: T2, T2b, T8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "image_id": p.image_id,
            "set_id": p.set_id.value,
            "entity_id": p.entity_id,
            "wikidata_id": p.wikidata_id,
            "file_title": p.file_title,
            "page_url": p.page_url,
            "url": p.url,
            "thumb_url": p.thumb_url,
            "width": p.width,
            "height": p.height,
            "licence": p.licence,
            "licence_url": p.licence_url,
            "author": p.author,
            "credit": p.credit,
            "taken": p.taken,
            "latitude": p.latitude,
            "longitude": p.longitude,
            "camera": p.camera,
            "source": p.source.value,
            "category": p.category,
            "view": p.view.value,
            "phash_distance": p.phash_distance,
        }
        for p in photos
    ]
    _write(pa.Table.from_pylist(rows, schema=COMMONS_SCHEMA), path)


def read_commons(path: Path) -> list[CommonsPhoto]:
    return [
        CommonsPhoto(
            image_id=str(row["image_id"]),
            set_id=CommonsSet(row["set_id"]),
            entity_id=str(row["entity_id"]),
            wikidata_id=str(row["wikidata_id"]),
            file_title=str(row["file_title"]),
            page_url=str(row["page_url"]),
            url=str(row["url"]),
            thumb_url=str(row["thumb_url"]),
            width=int(row["width"]),
            height=int(row["height"]),
            licence=str(row["licence"]),
            licence_url=str(row["licence_url"]),
            author=str(row["author"]),
            credit=str(row["credit"]),
            taken=str(row["taken"]),
            latitude=_optional_float(row["latitude"]),
            longitude=_optional_float(row["longitude"]),
            camera=str(row["camera"]),
            source=Source(row["source"]),
            category=str(row["category"]),
            view=View(row["view"]),
            phash_distance=None if row["phash_distance"] is None else int(row["phash_distance"]),
        )
        for row in _read(path, COMMONS_SCHEMA).to_pylist()
    ]


# pyarrow-stubs leave a few parameter types unknown; keep the ignores to these two lines.
def _write(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path)  # pyright: ignore[reportUnknownMemberType]


def _read(path: Path, schema: pa.Schema) -> pa.Table:
    return pq.read_table(path, schema=schema)  # pyright: ignore[reportUnknownMemberType]


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(v) for v in cast("list[object]", value))


def _optional(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
