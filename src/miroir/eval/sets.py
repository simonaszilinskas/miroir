"""Test set files (technical spec §7): one JSONL per half, and gallery lists.

Read a half only with `load_queries`. Its sealed half needs `allow_sealed=True`, which only
round-R4 code may pass (CODING_STANDARDS 19). `write_set` writes both halves; it never reads.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import TypeAdapter

from miroir.config import Half, Settings


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    set_id: str
    entity_id: str
    """Building or object the query image shows. Halves are split on it."""
    query_image_id: str
    expected_entity_ids: tuple[str, ...]
    """Right answers. Empty for T8, where the right answer is "not found"."""
    exclude_image_ids: tuple[str, ...]
    """Take these out of the gallery before searching this query."""
    scene_image_ids: tuple[str, ...] = ()
    """T3 only: the Jeu de Paume room photo(s) the artwork appears in."""


_QUERY = TypeAdapter(Query)


def write_set(
    settings: Settings, set_id: str, tune: Sequence[Query], sealed: Sequence[Query]
) -> None:
    halves: tuple[tuple[Half, Sequence[Query]], ...] = (("tune", tune), ("sealed", sealed))
    for half, queries in halves:
        if any(q.set_id != set_id for q in queries):
            msg = f"{set_id} {half}: a query belongs to another set"
            raise ValueError(msg)
        path = settings.testset_path(set_id, half, allow_sealed=True)  # writing, not reading
        _write_lines(path, (json.dumps(asdict(q), ensure_ascii=False) for q in queries))


def load_queries(
    settings: Settings, set_id: str, half: Half, *, allow_sealed: bool = False
) -> list[Query]:
    """The one way to read a test set. `allow_sealed=True` is for round R4 only."""
    path = settings.testset_path(set_id, half, allow_sealed=allow_sealed)
    with path.open(encoding="utf-8") as lines:
        return [_QUERY.validate_json(line) for line in lines if line.strip()]


def tune_query_images(settings: Settings, set_ids: Sequence[str]) -> tuple[list[str], str]:
    """The query image ids of the sets' tune halves, once each, and a run label naming the sets.

    One vector run then holds the queries of several sets, e.g. "T2-T2b-T8c-tune-queries".
    """
    if not set_ids or len(set(set_ids)) != len(set_ids):
        msg = f"give one or more distinct test sets, got {list(set_ids)}"
        raise ValueError(msg)
    queries = [q for s in set_ids for q in load_queries(settings, s, "tune")]
    ids = list(dict.fromkeys(q.query_image_id for q in queries))
    return ids, f"{'-'.join(set_ids)}-tune-queries"


def write_gallery(settings: Settings, name: str, image_ids: Iterable[str]) -> Path:
    path = settings.gallery_path(name)
    _write_lines(path, sorted(image_ids))
    return path


def read_gallery(settings: Settings, name: str) -> list[str]:
    text = settings.gallery_path(name).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line]


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for line in lines:
            out.write(line + "\n")
