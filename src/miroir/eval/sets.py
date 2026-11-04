"""Test set files (technical spec §7): one JSONL per half, and gallery lists.

Read a half only with `load_queries`. Its sealed half needs `allow_sealed=True`, which only
round-R4 code may pass (CODING_STANDARDS 19). `write_set` writes both halves; it never reads.
It also writes the set's entities, both halves mixed and no query (`entities_in_sets`), so
what learns from the gallery can leave every test entity out without opening a sealed half.
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
    text: str = ""
    """T7 only: the words searched. `query_image_id` is then the hidden image they describe."""
    kind: str = ""
    """PARTS and VET: what the whole is (altarpiece, statue, building...)."""
    why: str = ""
    """PARTS and VET: why the query counts as a part of the expected whole, and the target photo."""


_QUERY = TypeAdapter(Query)


def cluster_of(query: Query) -> str:
    """What the query is not independent of, for the bootstrap: its building or object (T2
    holds 1,000 photos of 385 buildings), or for T3 its room photo(s), which every query of
    the room is cropped from."""
    if query.scene_image_ids:
        return "scene:" + "+".join(query.scene_image_ids)
    return query.entity_id


def write_set(
    settings: Settings, set_id: str, tune: Sequence[Query], sealed: Sequence[Query]
) -> None:
    halves: tuple[tuple[Half, Sequence[Query]], ...] = (("tune", tune), ("sealed", sealed))
    for half, queries in halves:
        if any(q.set_id != set_id for q in queries):
            msg = f"{set_id} {half}: a query belongs to another set"
            raise ValueError(msg)
        path = settings.testset_path(set_id, half, allow_sealed=True)  # writing, not reading
        _write_lines(path, (json.dumps(_fields(q), ensure_ascii=False) for q in queries))
    used = {e for q in (*tune, *sealed) for e in (q.entity_id, *q.expected_entity_ids)}
    _write_lines(settings.testset_entities_path(set_id), sorted(used))


#: Fields only some sets fill: written only when set, so other sets' files stay as they were.
_OPTIONAL = ("text", "kind", "why")


def _fields(query: Query) -> dict[str, object]:
    fields = asdict(query)
    for name in _OPTIONAL:
        if not fields[name]:
            del fields[name]
    return fields


def load_queries(
    settings: Settings, set_id: str, half: Half, *, allow_sealed: bool = False
) -> list[Query]:
    """The one way to read a test set. `allow_sealed=True` is for round R4 only."""
    path = settings.testset_path(set_id, half, allow_sealed=allow_sealed)
    with path.open(encoding="utf-8") as lines:
        return [_QUERY.validate_json(line) for line in lines if line.strip()]


def entities_in_sets(settings: Settings) -> frozenset[str]:
    """Every building or object any test set uses, tune and sealed: a query's own entity and
    its answers. Read from the lists `write_set` keeps beside the tune halves; no sealed file
    is opened. Fails for a set written before the lists existed."""
    out: set[str] = set()
    for tune in sorted(settings.testsets_dir.glob("*.tune.jsonl")):
        set_id = tune.name.removesuffix(".tune.jsonl")
        path = settings.testset_entities_path(set_id)
        if not path.exists():
            msg = f"{set_id} has no entity list ({path}): rebuild it (`miroir testsets build`)"
            raise FileNotFoundError(msg)
        out.update(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    return frozenset(out)


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
