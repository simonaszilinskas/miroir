"""Build `records.parquet` and `images.parquet` from the POP exports."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from miroir.config import Settings
from miroir.records.mappers import RawRecord, map_record
from miroir.records.model import JULY_COLLECTIONS, Collection, Record
from miroir.records.store import write_records

logger = logging.getLogger(__name__)

#: Counts the build must reproduce on the July 2026 exports (technical spec §3).
#: Keys: (collection, counter). A mismatch is logged, not fatal: new exports differ.
EXPECTED: dict[tuple[Collection, str], int] = {
    (Collection.JOCONDE, "records_read"): 33_000,
    (Collection.JOCONDE, "records_kept"): 32_997,
    (Collection.JOCONDE, "images"): 54_825,
    (Collection.JOCONDE, "skip_record:no_image"): 3,
    (Collection.JOCONDE, "skip_image:placeholder"): 3,
    (Collection.JOCONDE, "skip_image:duplicate"): 1,
    (Collection.MEMOIRE, "records_read"): 66_000,
    (Collection.MEMOIRE, "records_kept"): 65_989,
    (Collection.MEMOIRE, "images"): 65_989,
    (Collection.MEMOIRE, "skip_record:no_image"): 11,
    (Collection.MEMOIRE, "skip_image:placeholder"): 1,
    (Collection.MEMOIRE, "skip_image:on_request"): 10,
    (Collection.ROSE_VALLAND, "records_read"): 2_400,
    (Collection.ROSE_VALLAND, "records_kept"): 2_400,
    (Collection.ROSE_VALLAND, "images"): 16_961,
    (Collection.ROSE_VALLAND, "skip_image:duplicate"): 8,
    (Collection.ROSE_VALLAND, "image_type:detail"): 1_351,
    (Collection.ROSE_VALLAND, "image_type:back"): 2_038,
    (Collection.ROSE_VALLAND, "image_type:inscription"): 1_457,
    (Collection.ROSE_VALLAND, "image_type:label"): 1_518,
    (Collection.ROSE_VALLAND, "image_type:main"): 10_597,
}


@dataclass(frozen=True, slots=True)
class Mismatch:
    collection: Collection
    counter: str
    expected: int
    actual: int


@dataclass(slots=True)
class BuildReport:
    counts: dict[Collection, Counter[str]] = field(default_factory=dict[Collection, Counter[str]])

    def mismatches(self) -> list[Mismatch]:
        out: list[Mismatch] = []
        for (collection, counter), expected in EXPECTED.items():
            if collection not in self.counts:
                continue
            actual = self.counts[collection][counter]
            if actual != expected:
                out.append(Mismatch(collection, counter, expected, actual))
        return out


def iter_raw(path: Path) -> Iterator[RawRecord]:
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                msg = f"{path}:{number}: expected a JSON object"
                raise TypeError(msg)
            yield cast("RawRecord", value)


def map_collection(collection: Collection, path: Path) -> tuple[list[Record], Counter[str]]:
    records: list[Record] = []
    counts: Counter[str] = Counter()
    for raw in iter_raw(path):
        counts["records_read"] += 1
        result = map_record(collection, raw)
        for skip in result.image_skips:
            counts[f"skip_image:{skip}"] += 1
        if result.record is None:
            counts[f"skip_record:{result.record_skip}"] += 1
            continue
        counts["records_kept"] += 1
        counts["images"] += len(result.record.images)
        for image in result.record.images:
            counts[f"image_type:{image.image_type}"] += 1
        records.append(result.record)
    return records, counts


def build(
    settings: Settings, collections: tuple[Collection, ...] = JULY_COLLECTIONS
) -> BuildReport:
    report = BuildReport()
    records: list[Record] = []
    for collection in collections:
        path = settings.export_jsonl(collection)
        logger.info("reading %s", path)
        mapped, counts = map_collection(collection, path)
        records.extend(mapped)
        report.counts[collection] = counts
        for counter, value in sorted(counts.items()):
            logger.info("%s %s=%d", collection, counter, value)

    _check_unique(records)
    write_records(records, settings.records_parquet, settings.images_parquet)
    logger.info("wrote %d records to %s", len(records), settings.records_parquet)
    logger.info(
        "wrote %d images to %s", sum(len(r.images) for r in records), settings.images_parquet
    )
    for m in report.mismatches():
        logger.warning("%s %s: expected %d, got %d", m.collection, m.counter, m.expected, m.actual)
    return report


def _check_unique(records: list[Record]) -> None:
    refs = Counter(r.ref for r in records)
    image_ids = Counter(i.image_id for r in records for i in r.images)
    dup_refs = [ref for ref, n in refs.items() if n > 1]
    dup_ids = [image_id for image_id, n in image_ids.items() if n > 1]
    if dup_refs or dup_ids:
        msg = f"duplicate refs {dup_refs[:5]} or image ids {dup_ids[:5]}"
        raise ValueError(msg)
