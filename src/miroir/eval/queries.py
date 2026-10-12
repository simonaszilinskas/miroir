"""Read a test set half (JSONL, technical spec §7) into image queries for the scorer.

Each line is one query. Keys the scorer uses:

- `query_image`: `image_id` of the query image (it must have a vector),
- `expected_entity_ids`: entities that count as a right answer,
- `excluded_image_ids` (optional): images hidden from the gallery for this query, e.g. the
  other images of the query's photo record,
- `query_id` (optional): defaults to `query_image`.

Lines with `query_text` and no `query_image` (word search, step B) or with a `box` (needs a
crop vector, step C) are not image-to-image queries: they are counted and skipped.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import cast

from miroir.eval.scoring import Query

logger = logging.getLogger(__name__)


def read_queries(path: Path) -> tuple[Query, ...]:
    queries: list[Query] = []
    skipped: Counter[str] = Counter()
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parsed: object = json.loads(line)
            if not isinstance(parsed, dict):
                msg = f"{path}:{number}: not a JSON object"
                raise ValueError(msg)
            row = cast("dict[str, object]", parsed)
            if "query_image" not in row:
                skipped["text query"] += 1
                continue
            if row.get("box") is not None:
                skipped["box query"] += 1
                continue
            queries.append(_query(row, f"{path}:{number}"))
    for reason, count in skipped.items():
        logger.info("%s: skipped %d %s(s)", path.name, count, reason)
    return tuple(queries)


def _query(row: dict[str, object], where: str) -> Query:
    image = row["query_image"]
    expected = row.get("expected_entity_ids")
    excluded = row.get("excluded_image_ids", [])
    query_id = row.get("query_id", image)
    if not isinstance(image, str) or not isinstance(query_id, str):
        msg = f"{where}: query_image and query_id must be strings"
        raise ValueError(msg)
    return Query(
        query_id=query_id,
        image_id=image,
        entity_ids=_strings(expected, where, "expected_entity_ids"),
        excluded_image_ids=_strings(excluded, where, "excluded_image_ids"),
    )


def _strings(value: object, where: str, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):  # pyright: ignore[reportUnknownVariableType]
        msg = f"{where}: {key} must be a list of strings"
        raise ValueError(msg)
    return tuple(str(v) for v in value)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
