"""Build test sets T1, T3, T5, T8 and the galleries from the record table (technical spec §7).

Every draw uses a `random.Random` seeded from `config.SEED` and the set id, over inputs sorted
by id, so the same records always give the same sets.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from miroir.config import SEED, Settings
from miroir.eval.sets import Query, write_gallery, write_set
from miroir.records.model import Collection, Record
from miroir.records.store import read_records

logger = logging.getLogger(__name__)

#: Queries per set as technical spec §7 asks. T3 takes every eligible artwork.
SPEC_SIZES: dict[str, int] = {"T1": 1_000, "T3": 231, "T5": 1_000, "T8": 500}

#: Gallery for comparing models before the winners run on everything.
SCREENING_SIZE = 20_000
SCREENING = "screening-20k"
SCREENING_PUBLISHABLE = "screening-20k-publishable"
FULL = "full"

#: Rose Valland refs of the Jeu de Paume room photos and the artworks seen in them.
JEU_DE_PAUME_PREFIX = "JDP-"


@dataclass(frozen=True, slots=True)
class Sizes:
    t1: int = SPEC_SIZES["T1"]
    t5: int = SPEC_SIZES["T5"]
    t8_per_part: int = SPEC_SIZES["T8"] // 4
    """T8 draws this many from each of T1 tune, T1 sealed, T5 tune, T5 sealed."""
    screening: int = SCREENING_SIZE


@dataclass(frozen=True, slots=True)
class SplitSet:
    set_id: str
    tune: tuple[Query, ...]
    sealed: tuple[Query, ...]

    @property
    def queries(self) -> tuple[Query, ...]:
        return (*self.tune, *self.sealed)


@dataclass(frozen=True, slots=True)
class SetReport:
    set_id: str
    tune: int
    sealed: int
    spec: int
    closed_ok: int
    """Queries a closed model may run: publishable query image and a publishable answer."""


@dataclass(frozen=True, slots=True)
class BuildReport:
    sets: tuple[SetReport, ...]
    full: int
    screening: int
    needed: int
    """Images some query needs (query, answers, room photos); the rest are random distractors."""
    screening_publishable: int


class RecordIndex:
    """Lookups over the record table."""

    def __init__(self, records: Sequence[Record]) -> None:
        self.by_ref = {r.ref: r for r in records}
        self.image_publishable = {i.image_id: r.publishable for r in records for i in r.images}
        entity_images: defaultdict[str, list[str]] = defaultdict(list)
        for record in records:
            entity_images[record.entity_id].extend(i.image_id for i in record.images)
        self.entity_images = {e: tuple(sorted(ids)) for e, ids in entity_images.items()}


def _rng(seed: int, set_id: str) -> random.Random:
    return random.Random(f"{seed}:{set_id}")


def _sample[T](rng: random.Random, items: Sequence[T], size: int, set_id: str) -> list[T]:
    if len(items) < size:
        logger.warning("%s: only %d eligible, wanted %d; taking all", set_id, len(items), size)
        return list(items)
    return rng.sample(items, size)


# --- the sets -------------------------------------------------------------------


def t1_queries(records: Sequence[Record], size: int, seed: int = SEED) -> list[Query]:
    """Mémoire buildings (PA) with 2+ photos. One photo is the query; the others are the answer."""
    rng = _rng(seed, "T1")
    photos: defaultdict[str, list[tuple[str, Record]]] = defaultdict(list)
    for record in records:
        if record.collection == Collection.MEMOIRE and record.entity_id.startswith("PA"):
            photos[record.entity_id].extend((i.image_id, record) for i in record.images)
    eligible = sorted(e for e, items in photos.items() if len(items) >= 2)
    logger.info("T1: %d buildings with 2+ photos", len(eligible))
    queries: list[Query] = []
    for entity in _sample(rng, eligible, size, "T1"):
        image_id, record = rng.choice(sorted(photos[entity], key=lambda item: item[0]))
        queries.append(
            Query(
                query_id=f"T1-{image_id}",
                set_id="T1",
                entity_id=entity,
                query_image_id=image_id,
                expected_entity_ids=(entity,),
                # The query's whole photo record leaves the gallery (one image in Mémoire).
                exclude_image_ids=tuple(sorted(i.image_id for i in record.images)),
            )
        )
    return queries


def t5_queries(records: Sequence[Record], size: int, seed: int = SEED) -> list[Query]:
    """Joconde records with 2+ images. One view is the query; the record's other views answer."""
    rng = _rng(seed, "T5")
    eligible = sorted(
        (r for r in records if r.collection == Collection.JOCONDE and len(r.images) >= 2),
        key=lambda r: r.ref,
    )
    logger.info("T5: %d Joconde records with 2+ images", len(eligible))
    queries: list[Query] = []
    for record in _sample(rng, eligible, size, "T5"):
        image_id = rng.choice(sorted(i.image_id for i in record.images))
        queries.append(
            Query(
                query_id=f"T5-{image_id}",
                set_id="T5",
                entity_id=record.entity_id,
                query_image_id=image_id,
                expected_entity_ids=(record.entity_id,),
                exclude_image_ids=(image_id,),  # the other views are the answer: they stay
            )
        )
    return queries


def t3_queries(records: Sequence[Record], seed: int = SEED) -> list[Query]:
    """Jeu de Paume artworks. No hand-drawn boxes exist, so the query is the artwork's own crop.

    Each artwork record holds 1-3 crops of the room photo(s) it hangs in (the detail as seen,
    a rectified version, sometimes a closer detail). One crop is the query, standing in for a
    perfect box; the record's other crops are the answer. Artworks with one crop are left out.
    The room photos go in `scene_image_ids` so step C can be scored end to end on them.
    """
    rng = _rng(seed, "T3")
    jdp = {r.ref: r for r in records if r.collection == Collection.ROSE_VALLAND}
    jdp = {ref: r for ref, r in jdp.items() if ref.startswith(JEU_DE_PAUME_PREFIX)}
    artworks = sorted((r for r in jdp.values() if r.domain), key=lambda r: r.ref)
    queries: list[Query] = []
    skipped = 0
    for artwork in artworks:
        scenes = [jdp[ref] for ref in artwork.linked_refs if ref in jdp and not jdp[ref].domain]
        if len(artwork.images) < 2 or not scenes:
            skipped += 1
            continue
        image_id = rng.choice(sorted(i.image_id for i in artwork.images))
        queries.append(
            Query(
                query_id=f"T3-{image_id}",
                set_id="T3",
                entity_id=artwork.entity_id,
                query_image_id=image_id,
                expected_entity_ids=(artwork.entity_id,),
                exclude_image_ids=(image_id,),
                scene_image_ids=tuple(sorted(i.image_id for s in scenes for i in s.images)),
            )
        )
    logger.info(
        "T3: %d Jeu de Paume artworks, %d with 2+ crops and a room photo, %d skipped",
        len(artworks),
        len(queries),
        skipped,
    )
    return queries


def t8_split(
    sources: Sequence[SplitSet], index: RecordIndex, per_part: int, seed: int = SEED
) -> SplitSet:
    """Queries from T1 and T5 with their whole entity taken out of the gallery.

    Each T8 query keeps the half of the query it comes from, so halves stay apart across sets.
    """
    rng = _rng(seed, "T8")
    halves: dict[str, list[Query]] = {"tune": [], "sealed": []}
    for source in sources:
        for half, queries in (("tune", source.tune), ("sealed", source.sealed)):
            ordered = sorted(queries, key=lambda q: q.query_id)
            for q in _sample(rng, ordered, per_part, f"T8 from {source.set_id} {half}"):
                halves[half].append(
                    Query(
                        query_id=f"T8-{q.query_image_id}",
                        set_id="T8",
                        entity_id=q.entity_id,
                        query_image_id=q.query_image_id,
                        expected_entity_ids=(),
                        exclude_image_ids=index.entity_images[q.entity_id],
                    )
                )
    return SplitSet("T8", tuple(halves["tune"]), tuple(halves["sealed"]))


# --- halves and galleries -------------------------------------------------------


def split_by_entity(set_id: str, queries: Iterable[Query], seed: int = SEED) -> SplitSet:
    """Half the entities (rounded up) go to `tune`, the rest to `sealed`."""
    queries = sorted(queries, key=lambda q: q.query_id)
    entities = sorted({q.entity_id for q in queries})
    _rng(seed, f"{set_id}:split").shuffle(entities)
    tune_entities = set(entities[: (len(entities) + 1) // 2])
    return SplitSet(
        set_id,
        tune=tuple(q for q in queries if q.entity_id in tune_entities),
        sealed=tuple(q for q in queries if q.entity_id not in tune_entities),
    )


def check_halves_apart(splits: Iterable[SplitSet]) -> None:
    """Raise if any entity is in a tune half and a sealed half, across every set."""
    tune: set[str] = set()
    sealed: set[str] = set()
    for split in splits:
        tune.update(q.entity_id for q in split.tune)
        sealed.update(q.entity_id for q in split.sealed)
    both = sorted(tune & sealed)
    if both:
        msg = f"{len(both)} entities are in both halves, e.g. {both[:5]}"
        raise ValueError(msg)


def needed_images(queries: Iterable[Query], index: RecordIndex) -> set[str]:
    """Every image a query uses: the query, all images of its answers, its room photos."""
    needed: set[str] = set()
    for q in queries:
        needed.add(q.query_image_id)
        needed.update(q.scene_image_ids)
        for entity in q.expected_entity_ids:
            needed.update(index.entity_images[entity])
    return needed


def screening_gallery(
    all_image_ids: Iterable[str], needed: set[str], size: int, seed: int = SEED
) -> set[str]:
    """`needed` plus seeded random distractors, `size` images in all."""
    if len(needed) > size:
        msg = f"queries need {len(needed)} images, more than the {size}-image screening gallery"
        raise ValueError(msg)
    rest = sorted(set(all_image_ids) - needed)
    return needed | set(_rng(seed, "screening").sample(rest, size - len(needed)))


def closed_ok(query: Query, publishable_gallery: set[str], index: RecordIndex) -> bool:
    """A closed model may run the query: its image is publishable, and so is one answer image."""
    if not index.image_publishable[query.query_image_id]:
        return False
    answers = (i for e in query.expected_entity_ids for i in index.entity_images[e])
    return not query.expected_entity_ids or any(i in publishable_gallery for i in answers)


# --- the whole build ----------------------------------------------------------------


def build_sets(records: Sequence[Record], sizes: Sizes, seed: int = SEED) -> list[SplitSet]:
    index = RecordIndex(records)
    t1 = split_by_entity("T1", t1_queries(records, sizes.t1, seed), seed)
    t3 = split_by_entity("T3", t3_queries(records, seed), seed)
    t5 = split_by_entity("T5", t5_queries(records, sizes.t5, seed), seed)
    t8 = t8_split((t1, t5), index, sizes.t8_per_part, seed)
    splits = [t1, t3, t5, t8]
    check_halves_apart(splits)
    return splits


def build(settings: Settings, sizes: Sizes | None = None, seed: int = SEED) -> BuildReport:
    sizes = sizes or Sizes()
    records = read_records(settings.records_parquet, settings.images_parquet)
    index = RecordIndex(records)
    splits = build_sets(records, sizes, seed)
    for split in splits:
        write_set(settings, split.set_id, split.tune, split.sealed)

    all_ids = set(index.image_publishable)
    needed = needed_images((q for s in splits for q in s.queries), index)
    screening = screening_gallery(all_ids, needed, sizes.screening, seed)
    publishable = {i for i in screening if index.image_publishable[i]}
    write_gallery(settings, FULL, all_ids)
    write_gallery(settings, SCREENING, screening)
    write_gallery(settings, SCREENING_PUBLISHABLE, publishable)

    reports = tuple(
        SetReport(
            set_id=s.set_id,
            tune=len(s.tune),
            sealed=len(s.sealed),
            spec=SPEC_SIZES[s.set_id],
            closed_ok=sum(closed_ok(q, publishable, index) for q in s.queries),
        )
        for s in splits
    )
    for r in reports:
        logger.info("%s: %d tune, %d sealed (spec %d)", r.set_id, r.tune, r.sealed, r.spec)
        if r.tune + r.sealed != r.spec:
            logger.warning(
                "%s has %d queries; spec §7 says %d", r.set_id, r.tune + r.sealed, r.spec
            )
    return BuildReport(
        sets=reports,
        full=len(all_ids),
        screening=len(screening),
        needed=len(needed),
        screening_publishable=len(publishable),
    )
