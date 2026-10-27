"""Records copied from the whole POP download into the site's index, one entity at a time.

The July tables stay as they are. `extra/` holds the copied records with their images, as
`miroir records build-full` mapped them; the index build and the privacy check read both
(`read_site_records`). A July record wins over an extra record with the same ref.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from miroir.config import Settings
from miroir.records.model import Image, Record
from miroir.records.store import read_images, read_records

logger = logging.getLogger(__name__)


def add_entities(settings: Settings, entity_ids: Sequence[str]) -> int:
    """Copy every record of `entity_ids` and its images from the full tables into `extra/`.

    Entities already there are replaced. Returns how many records `extra/` now holds.
    """
    wanted = sorted(set(entity_ids))
    pairs = (
        (settings.full_records_parquet, settings.extra_records_parquet),
        (settings.full_images_parquet, settings.extra_images_parquet),
    )
    held = 0
    for source, target in pairs:
        found = _read(source, "in", wanted)
        if found.num_rows == 0:
            msg = f"no row of {', '.join(wanted)} in {source}"
            raise ValueError(msg)
        if target.exists():
            kept = _read(target, "not in", wanted)
            found = pa.concat_tables([kept, found.cast(kept.schema)])
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(found, target)  # pyright: ignore[reportUnknownMemberType]
        logger.info("%s: %d rows", target, found.num_rows)
        if source == settings.full_records_parquet:
            held = found.num_rows
    return held


def _read(path: Path, op: Literal["in", "not in"], entity_ids: list[str]) -> pa.Table:
    """Rows whose entity is (`in`) or is not (`not in`) one of `entity_ids`."""
    return pq.read_table(path, filters=[("entity_id", op, entity_ids)])  # pyright: ignore[reportUnknownMemberType]


def read_extra_images(settings: Settings) -> list[Image]:
    if not settings.extra_images_parquet.exists():
        return []
    return read_images(settings.extra_images_parquet)


def read_site_records(settings: Settings) -> list[Record]:
    """The July records, then the extra records whose ref July does not hold."""
    records = read_records(settings.records_parquet, settings.images_parquet)
    if not settings.extra_records_parquet.exists():
        return records
    refs = {r.ref for r in records}
    extra = read_records(settings.extra_records_parquet, settings.extra_images_parquet)
    added = [r for r in extra if r.ref not in refs]
    logger.info("extra records: %d added to %d July records", len(added), len(records))
    return [*records, *added]
