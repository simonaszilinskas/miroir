"""Read and write `records.parquet` and `images.parquet`, and the other tables of the data dir.

Downstream code loads records with `read_records()` / `read_images()` rather than
opening the Parquet files itself, so the schema lives in one place.

Every table is written to a temporary file and renamed when complete, so a failed build leaves
the last good table. Each carries `SCHEMA_VERSION` in its metadata; a reader refuses a table of
another version or with other columns.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection as Many
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from miroir.commons.model import CommonsPhoto, CommonsSet, Source, View
from miroir.geo.model import Place, PlaceSource, PositionSource, QueryPosition
from miroir.records.model import Collection, Image, ImageType, Record

_STRINGS = pa.list_(pa.string())

#: Parquet metadata key holding the version of this module's schemas.
SCHEMA_VERSION_KEY = b"miroir.schema_version"
#: Bump on any change of columns or of what a value means. 1: image ids hash the bucket key
#: (`records.images.make_image_id`). Tables written before versions existed carry none; they
#: read if their columns match, and their image ids hash the URL (the July scheme).
SCHEMA_VERSION = b"1"


def _versioned(schema: pa.Schema) -> pa.Schema:
    return schema.with_metadata({SCHEMA_VERSION_KEY: SCHEMA_VERSION})  # pyright: ignore[reportUnknownMemberType]


RECORDS_SCHEMA = _versioned(
    pa.schema(
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
)

IMAGES_SCHEMA = _versioned(
    pa.schema(
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
)


COMMONS_SCHEMA = _versioned(
    pa.schema(
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
)

PLACES_SCHEMA = _versioned(
    pa.schema(
        [
            pa.field("entity_id", pa.string(), nullable=False),
            pa.field("latitude", pa.float64(), nullable=False),
            pa.field("longitude", pa.float64(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
        ]
    )
)

POSITIONS_SCHEMA = _versioned(
    pa.schema(
        [
            pa.field("image_id", pa.string(), nullable=False),
            pa.field("latitude", pa.float64(), nullable=False),
            pa.field("longitude", pa.float64(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
        ]
    )
)


def write_records(records: Sequence[Record], records_path: Path, images_path: Path) -> None:
    """Write both files. Rows keep the order given."""
    with RecordsWriter(records_path, images_path) as writer:
        writer.write(records)


class RecordsWriter:
    """Write both files batch by batch, for exports too big to hold in memory (spec §3.4).

    Use it as a context manager: both files appear when the block ends without error; on an
    error neither changes.
    """

    def __init__(self, records_path: Path, images_path: Path) -> None:
        self._paths = (records_path, images_path)
        self._temps = (_temp(records_path), _temp(images_path))
        for path in self._temps:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._records = pq.ParquetWriter(self._temps[0], RECORDS_SCHEMA)
        self._images = pq.ParquetWriter(self._temps[1], IMAGES_SCHEMA)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self._records.close()
        self._images.close()
        if error is not None:
            for temp in self._temps:
                temp.unlink(missing_ok=True)
            return
        for temp, path in zip(self._temps, self._paths, strict=True):
            temp.replace(path)

    def write(self, records: Sequence[Record]) -> None:
        if records:
            self._records.write_table(_records_table(records))
            self._images.write_table(_images_table(records))


def _records_table(records: Sequence[Record]) -> pa.Table:
    rows = [
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
    return pa.Table.from_pylist(rows, schema=RECORDS_SCHEMA)


def _images_table(records: Sequence[Record]) -> pa.Table:
    rows = [
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
    return pa.Table.from_pylist(rows, schema=IMAGES_SCHEMA)


def read_column(path: Path, name: str) -> list[object]:
    """One column of a records or images file, as Python values (for counts over big tables)."""
    schema = _file_schema(path)
    _check(path, schema, RECORDS_SCHEMA if "publishable" in schema.names else IMAGES_SCHEMA)
    table = pq.read_table(path, columns=[name])  # pyright: ignore[reportUnknownMemberType]
    values: list[object] = list(table.column(name).to_pylist())
    return values


def read_images(images_path: Path, entity_ids: Many[str] | None = None) -> list[Image]:
    """Every image, or only those of `entity_ids` (read with a filter, for the full tables)."""
    rows = _read(images_path, IMAGES_SCHEMA, entity_ids).to_pylist()
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


def read_records(
    records_path: Path, images_path: Path, entity_ids: Many[str] | None = None
) -> list[Record]:
    """Load every record with its images, in file order; or only the records of `entity_ids`."""
    by_ref: defaultdict[str, list[Image]] = defaultdict(list)
    for image in read_images(images_path, entity_ids):
        by_ref[image.ref].append(image)
    rows = _read(records_path, RECORDS_SCHEMA, entity_ids).to_pylist()
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


def write_places(places: Sequence[Place], path: Path) -> None:
    """One position per entity (step H2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "entity_id": p.entity_id,
            "latitude": p.latitude,
            "longitude": p.longitude,
            "source": p.source.value,
        }
        for p in places
    ]
    _write(pa.Table.from_pylist(rows, schema=PLACES_SCHEMA), path)


def read_places(path: Path) -> list[Place]:
    return [
        Place(
            entity_id=str(row["entity_id"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            source=PlaceSource(row["source"]),
        )
        for row in _read(path, PLACES_SCHEMA).to_pylist()
    ]


def write_positions(positions: Sequence[QueryPosition], path: Path) -> None:
    """Where each query photo was taken, when known (step H2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "image_id": p.image_id,
            "latitude": p.latitude,
            "longitude": p.longitude,
            "source": p.source.value,
        }
        for p in positions
    ]
    _write(pa.Table.from_pylist(rows, schema=POSITIONS_SCHEMA), path)


def read_positions(path: Path) -> list[QueryPosition]:
    return [
        QueryPosition(
            image_id=str(row["image_id"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            source=PositionSource(row["source"]),
        )
        for row in _read(path, POSITIONS_SCHEMA).to_pylist()
    ]


def write_table(table: pa.Table, path: Path) -> None:
    """Write a table whole or not at all: to a temporary file, renamed when complete."""
    temp = _temp(path)
    temp.parent.mkdir(parents=True, exist_ok=True)
    try:
        pq.write_table(table, temp)  # pyright: ignore[reportUnknownMemberType]
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    temp.replace(path)


def _temp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _write(table: pa.Table, path: Path) -> None:
    write_table(table, path)


def _check(path: Path, found: pa.Schema, schema: pa.Schema) -> None:
    """Refuse a table of another schema version, or whose columns differ from `schema`."""
    version = (found.metadata or {}).get(SCHEMA_VERSION_KEY)
    if version is not None and version != SCHEMA_VERSION:
        msg = (
            f"{path}: schema version {version.decode()}, this miroir reads "
            f"{SCHEMA_VERSION.decode()}; rebuild it"
        )
        raise ValueError(msg)
    if not found.remove_metadata().equals(schema.remove_metadata()):
        msg = f"{path}: columns differ from this miroir's schema; rebuild it"
        raise ValueError(msg)


# pyarrow-stubs leave a few parameter types unknown; keep the ignores to these lines.
def _file_schema(path: Path) -> pa.Schema:
    return pq.read_schema(path)  # pyright: ignore[reportUnknownMemberType]


def _read(path: Path, schema: pa.Schema, entity_ids: Many[str] | None = None) -> pa.Table:
    _check(path, _file_schema(path), schema)
    if entity_ids is None:
        return pq.read_table(path, schema=schema)  # pyright: ignore[reportUnknownMemberType]
    wanted = pc.field("entity_id").isin(sorted(entity_ids))  # pyright: ignore[reportUnknownMemberType]
    return pq.read_table(path, schema=schema, filters=wanted)  # pyright: ignore[reportUnknownMemberType]


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(v) for v in cast("list[object]", value))


def _optional(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
