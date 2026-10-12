"""A sample of Commons queries for a person to check by eye: is the answer right?"""

from __future__ import annotations

import json
import random
from pathlib import Path

from miroir.config import SEED, Settings
from miroir.eval.sets import load_queries
from miroir.records.model import Record
from miroir.records.store import read_commons, read_records

#: Queries in the sample.
CHECK_SIZE = 100


def check_sample(
    settings: Settings, size: int = CHECK_SIZE, seed: int = SEED
) -> list[dict[str, str]]:
    """`size` seeded tune-half T2 and T2b queries: the Commons photo and one image of the answer."""
    photos = {p.image_id: p for p in read_commons(settings.commons_parquet)}
    records = sorted(
        read_records(settings.records_parquet, settings.images_parquet), key=lambda r: r.ref
    )
    by_entity: dict[str, Record] = {}
    for record in records:
        by_entity.setdefault(record.entity_id, record)
    queries = [q for s in ("T2", "T2b") for q in load_queries(settings, s, "tune")]
    chosen = random.Random(f"{seed}:commons-check").sample(queries, min(size, len(queries)))
    sample: list[dict[str, str]] = []
    for q in sorted(chosen, key=lambda q: q.query_id):
        photo = photos[q.query_image_id]
        record = by_entity[q.entity_id]
        sample.append(
            {
                "query_id": q.query_id,
                "set_id": q.set_id,
                "entity_id": q.entity_id,
                "commons_thumb_url": photo.thumb_url,
                "commons_page_url": photo.page_url,
                "commons_licence": photo.licence,
                "commons_author": photo.author,
                "view": photo.view.value,
                "pop_title": record.title or ", ".join(record.denomination),
                "pop_url": record.pop_url,
                "answer_image_url": record.cover.url,
            }
        )
    return sample


def write_check_sample(settings: Settings, path: Path, size: int = CHECK_SIZE) -> int:
    sample = check_sample(settings, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return len(sample)
