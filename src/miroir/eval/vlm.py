"""Step H1's text side: a vision model's answer about a query photo, searched with BM25.

The answer (`miroir.jobs.describe`) becomes a text query (`query_text`), searched over the
gallery's records (`text_gallery`); a record's score goes to each of its gallery images.
Step H4 replays H1 and H3 with these (`miroir.eval.h4`). The H1 and H3 scoring command is
retired (docs/experiments.md, "Retired code").
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from miroir.eval.bm25 import Bm25, record_text
from miroir.jobs.describe import Description
from miroir.records.model import Record

logger = logging.getLogger(__name__)

#: Image-model weights tried for H3; the text side gets the rest.
H3_WEIGHT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def query_text(description: Description | None, *, visible_text: bool = True) -> str:
    """The text searched for one answer. Name and commune count twice: they pin a record."""
    if description is None:
        return ""
    d = description
    parts = [d.name, d.name, d.kind, d.commune, d.commune, d.departement, d.region, d.period]
    parts += [d.author, d.description]
    if visible_text:
        parts.append(d.visible_text)
    return " ".join(p for p in parts if p)


@dataclass(frozen=True, slots=True)
class TextGallery:
    """The gallery's images, each with its entity and the row of its record in the index."""

    image_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    record_rows: np.ndarray[tuple[int], np.dtype[np.int64]]
    index: Bm25


def text_gallery(records: Sequence[Record], image_ids: Sequence[str]) -> TextGallery:
    """BM25 over the records that have an image in `image_ids`."""
    record_of = {i.image_id: r for r in records for i in r.images}
    image_of = {i.image_id: i for r in records for i in r.images}
    missing = [i for i in image_ids if i not in record_of]
    if missing:
        msg = f"{len(missing)} gallery image(s) have no record, e.g. {missing[:3]}"
        raise ValueError(msg)
    row_of: dict[str, int] = {}
    texts: list[str] = []
    for image_id in image_ids:
        record = record_of[image_id]
        if record.ref not in row_of:
            row_of[record.ref] = len(texts)
            texts.append(record_text(record))
    rows = np.fromiter((row_of[record_of[i].ref] for i in image_ids), dtype=np.int64)
    entities = tuple(image_of[i].entity_id for i in image_ids)
    logger.info("BM25 over %d records (%d images)", len(texts), len(image_ids))
    return TextGallery(tuple(image_ids), entities, rows, Bm25(texts))
