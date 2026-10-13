"""Tables of the whole POP download (technical spec §3.4 and §11).

`build_full` maps every base of `exports-full/` into `full-pop/records.parquet` and
`full-pop/images.parquet`, the same schema as the July tables, batch by batch so memory stays
small. `count_full` then reads a few columns back and counts what an embedding run would
see: distinct image URLs (the same Mémoire photo is listed by its Mémoire record and by the
Mérimée or Palissy record it shows), record covers, buildings and objects. It writes
`report.json` and one image list per option of §11.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from miroir.config import Settings
from miroir.records.build import iter_raw
from miroir.records.mappers import (
    FIELD_MAPS,
    FieldMap,
    Rights,
    full_field_maps,
    joconde_open,
    map_record,
    record_ref,
)
from miroir.records.model import Collection, Record
from miroir.records.store import RecordsWriter, read_column

logger = logging.getLogger(__name__)

#: Mémoire first: Mérimée and Palissy rights depend on its records, and a photo listed by both
#: keeps the Mémoire row as its first, so its own caption and photographer.
FULL_ORDER = (
    Collection.MEMOIRE,
    Collection.JOCONDE,
    Collection.MERIMEE,
    Collection.PALISSY,
    Collection.ROSE_VALLAND,
    Collection.MUSEO,
)
BATCH_RECORDS = 20_000

#: What each base's records show. Mémoire photos take the kind of the record they show.
BUILDING_BASES = ("merimee", "museo")
OBJECT_BASES = ("joconde", "palissy", "rose-valland")
KINDS = ("unlinked", "building", "object")
"""Index = kind code. `unlinked`: a Mémoire photo whose `LBASE` names no PA or PM record."""


@dataclass(frozen=True, slots=True)
class FullBuild:
    """Build counters per base: records read and kept, images, skips by reason."""

    counts: dict[Collection, Counter[str]] = field(default_factory=dict[Collection, Counter[str]])


def build_full(settings: Settings) -> FullBuild:
    """Map every base of the full export into the `full-pop` tables."""
    report = FullBuild()
    joconde = frozenset(
        record_ref(raw)
        for raw in iter_raw(settings.export_jsonl(Collection.JOCONDE))
        if joconde_open(raw)
    )
    logger.info("July Joconde: %d records publishable", len(joconde))
    memoire_open: set[str] = set()
    fmaps: dict[Collection, FieldMap] = {Collection.MEMOIRE: FIELD_MAPS[Collection.MEMOIRE]}
    writer = RecordsWriter(settings.full_records_parquet, settings.full_images_parquet)
    try:
        for collection in FULL_ORDER:
            if collection not in fmaps:
                rights = Rights(joconde_open=joconde, memoire_open=frozenset(memoire_open))
                fmaps = full_field_maps(rights)
            counts = _map_base(settings, collection, fmaps[collection], writer, memoire_open)
            report.counts[collection] = counts
            for counter, value in sorted(counts.items()):
                logger.info("%s %s=%d", collection, counter, value)
    finally:
        writer.close()
    return report


def _map_base(
    settings: Settings,
    collection: Collection,
    fmap: FieldMap,
    writer: RecordsWriter,
    memoire_open: set[str],
) -> Counter[str]:
    path = settings.full_export_jsonl(collection.value)
    logger.info("reading %s", path)
    counts: Counter[str] = Counter()
    batch: list[Record] = []
    for raw in iter_raw(path):
        counts["records_read"] += 1
        result = map_record(collection, raw, fmap)
        for skip in result.image_skips:
            counts[f"skip_image:{skip}"] += 1
        record = result.record
        if record is None:
            counts[f"skip_record:{result.record_skip}"] += 1
            continue
        counts["records_kept"] += 1
        counts["images"] += len(record.images)
        counts["publishable"] += record.publishable
        if collection is Collection.MEMOIRE and record.publishable:
            memoire_open.add(record.ref)
        batch.append(record)
        if len(batch) >= BATCH_RECORDS:
            writer.write(batch)
            batch = []
    writer.write(batch)
    return counts


def count_full(settings: Settings, build: FullBuild | None = None) -> dict[str, object]:
    """Count distinct images, covers, buildings and objects; write `report.json` and lists.

    Holds a few columns of the tables in memory: about 2 GB for 5 million image rows.
    """
    images_path, records_path = settings.full_images_parquet, settings.full_records_parquet
    rec_refs = _objects(read_column(records_path, "ref"))
    if np.unique(rec_refs).size != rec_refs.size:
        msg = "a ref appears in more than one record of the full tables"
        raise ValueError(msg)
    rec_open = np.asarray(read_column(records_path, "publishable"), dtype=np.bool_)
    rec_entities = _objects(read_column(records_path, "entity_id"))

    image_ids = _objects(read_column(images_path, "image_id"))
    refs = _objects(read_column(images_path, "ref"))
    entities = _objects(read_column(images_path, "entity_id"))
    collections = _objects(read_column(images_path, "collection"))
    is_cover = np.asarray(read_column(images_path, "is_cover"), dtype=np.bool_)
    urls = _objects(read_column(images_path, "url"))
    logger.info("read %d image rows, %d records", urls.size, rec_refs.size)

    order = np.argsort(rec_refs)
    row_open = rec_open[order][np.searchsorted(rec_refs[order], refs)]
    kind = _kinds(collections, entities)

    unique_urls, first, inverse = np.unique(urls, return_index=True, return_inverse=True)
    count = unique_urls.size
    url_kind = np.zeros(count, dtype=np.int8)
    np.maximum.at(url_kind, inverse, kind)
    url_cover = np.bincount(inverse, weights=is_cover, minlength=count) > 0
    url_open = np.bincount(inverse, weights=~row_open, minlength=count) == 0
    base_codes = np.unique(collections, return_inverse=True)[1]
    pairs = np.unique(inverse.astype(np.int64) * 16 + base_codes)
    url_bases = np.bincount(pairs // 16, minlength=count)
    first_ids = image_ids[first]

    entity_names, entity_inverse = np.unique(entities, return_inverse=True)
    entity_kind = np.zeros(entity_names.size, dtype=np.int8)
    np.maximum.at(entity_kind, entity_inverse, kind)
    entity_url = inverse[_entity_cover_rows(entities, refs, is_cover, entity_inverse)]
    entity_urls = np.unique(entity_url)

    lists = {
        "unique": first_ids,
        "covers": first_ids[url_cover],
        "buildings": first_ids[url_kind == 1],
        "objects": first_ids[url_kind == 2],
        "unlinked": first_ids[url_kind == 0],
        "building-covers": first_ids[url_cover & (url_kind == 1)],
        "object-covers": first_ids[url_cover & (url_kind == 2)],
        "entity-covers": first_ids[entity_urls],
    }
    for name, ids in lists.items():
        path = settings.full_image_list(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")

    lower = np.unique(_objects([str(u).lower() for u in unique_urls])).size
    report: dict[str, object] = {
        "bases": _bases(settings, build),
        "records": int(rec_refs.size),
        "records_publishable": int(rec_open.sum()),
        "image_rows": int(urls.size),
        "unique_images": int(count),
        "unique_images_ignoring_case": int(lower),
        "unique_images_in_several_bases": int((url_bases > 1).sum()),
        "unique_images_first_listed_by": _value_counts(collections[first]),
        "unique_images_by_kind": _by_kind(url_kind),
        "unique_covers": int(url_cover.sum()),
        "unique_covers_by_kind": _by_kind(url_kind[url_cover]),
        "unique_images_publishable": int(url_open.sum()),
        "entities": int(np.unique(rec_entities).size),
        "entities_with_image_by_kind": _by_kind(entity_kind),
        "unique_entity_covers": int(entity_urls.size),
        "unique_entity_covers_by_kind": _by_kind(url_kind[entity_urls]),
    }
    settings.full_report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def summary_lines(report: dict[str, object]) -> list[str]:
    """The report's headline numbers, one per line."""
    return [
        f"{key}: {json.dumps(value, ensure_ascii=False)}"
        for key, value in report.items()
        if key != "bases"
    ]


def _entity_cover_rows(
    entities: NDArray[np.object_],
    refs: NDArray[np.object_],
    is_cover: NDArray[np.bool_],
    entity_inverse: NDArray[np.intp],
) -> NDArray[np.int64]:
    """One image row per building or object: the cover of its own record (the Mérimée,
    Palissy or Joconde record whose ref is the entity id), else its first image row."""
    rows = entities.size
    count = int(entity_inverse.max()) + 1 if rows else 0
    own = np.flatnonzero(is_cover & (refs == entities))
    best_own = np.full(count, rows, dtype=np.int64)
    np.minimum.at(best_own, entity_inverse[own], own)
    best_any = np.full(count, rows, dtype=np.int64)
    np.minimum.at(best_any, entity_inverse, np.arange(rows, dtype=np.int64))
    return np.where(best_own < rows, best_own, best_any)


def _objects(values: list[object]) -> NDArray[np.object_]:
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


def _bases(settings: Settings, build: FullBuild | None) -> object:
    """Per-base build counters: from this build, else from the last report."""
    if build is not None:
        return {c.value: dict(sorted(n.items())) for c, n in build.counts.items()}
    if settings.full_report_json.exists():
        old: dict[str, object] = json.loads(settings.full_report_json.read_text(encoding="utf-8"))
        return old.get("bases", {})
    return {}


def _kinds(collections: NDArray[np.object_], entities: NDArray[np.object_]) -> NDArray[np.int8]:
    """Kind code per image row (index into `KINDS`)."""
    kind = np.zeros(collections.size, dtype=np.int8)
    for row, (base, entity) in enumerate(zip(collections, entities, strict=True)):
        if base in OBJECT_BASES or (base == "memoire" and str(entity).startswith("PM")):
            kind[row] = 2
        elif base in BUILDING_BASES or (base == "memoire" and str(entity).startswith("PA")):
            kind[row] = 1
    return kind


def _by_kind(kind: NDArray[np.int8]) -> dict[str, int]:
    counts = np.bincount(kind, minlength=len(KINDS))
    return {name: int(counts[code]) for code, name in enumerate(KINDS)}


def _value_counts(values: NDArray[np.object_]) -> dict[str, int]:
    names, counts = np.unique(values, return_counts=True)
    return {str(n): int(c) for n, c in zip(names, counts, strict=True)}
