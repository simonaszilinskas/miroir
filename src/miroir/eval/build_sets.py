"""Build test sets T1, T3, T5, T8 and the galleries from the record table (technical spec §7).

When `commons.parquet` exists (`miroir testsets commons`), also T2, T2b and the Commons part
of T8, and the `screening-v2` galleries that hold their answers.

Every draw uses a `random.Random` seeded from `config.SEED` and the set id, over inputs sorted
by id, so the same records always give the same sets.
"""

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from miroir.commons.model import CommonsPhoto, CommonsSet, is_commons_image
from miroir.config import SEED, Half, Settings
from miroir.eval.sets import Query, write_gallery, write_set
from miroir.records.model import Collection, Record
from miroir.records.store import read_commons, read_records

logger = logging.getLogger(__name__)

#: Queries drawn per set in technical spec §7, before T1 drops buildings with one shoot only.
#: T3 takes every artwork with 2+ crops (231 planned).
SPEC_SIZES: dict[str, int] = {"T1": 1_000, "T3": 198, "T5": 1_000, "T8": 500}
#: Commons sets: at least this many queries (their size depends on what Commons holds).
MIN_SIZES: dict[str, int] = {"T2": 1_000, "T2b": 500}
#: Commons photos added to T8 (`miroir.commons.build.Targets`: 200 buildings, 100 works).
T8_COMMONS = 300

#: Gallery for comparing models before the winners run on everything.
SCREENING_SIZE = 20_000
SCREENING = "screening-20k"
SCREENING_PUBLISHABLE = "screening-20k-publishable"
FULL = "full"
#: `screening-20k` plus every answer image of T2 and T2b. Never holds a Commons photo.
SCREENING_V2 = "screening-v2"
SCREENING_V2_PUBLISHABLE = "screening-v2-publishable"
#: Suffix of the list of images a v2 gallery adds to the first one: only these need embedding.
DELTA = "-delta"

#: Rose Valland refs of the Jeu de Paume room photos and the artworks seen in them.
JEU_DE_PAUME_PREFIX = "JDP-"

#: Two Mémoire photos whose refs differ only in their last number, by at most this much, count
#: as one shoot. Among photos of one building, 95% of pairs within 50 share photographer and
#: date; within 100, 85%; within 200, 68%.
SHOOT_REF_WINDOW = 50

_REF_NUMBER = re.compile(r"(.*\D)?(\d+)(\D*)")


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
    dropped: int = 0
    """Queries drawn but left with no answer once their exclusions apply (T1's shoot rule)."""
    at_least: bool = False
    """`spec` is a minimum (Commons sets), not an exact size."""

    @property
    def meets_spec(self) -> bool:
        total = self.tune + self.sealed
        return total >= self.spec if self.at_least else total + self.dropped == self.spec


@dataclass(frozen=True, slots=True)
class BuildReport:
    sets: tuple[SetReport, ...]
    full: int
    screening: int
    needed: int
    """Images some query needs (query, answers, room photos); the rest are random distractors."""
    screening_publishable: int
    screening_v2: int = 0
    """0 when there are no Commons sets."""
    screening_v2_publishable: int = 0
    delta: int = 0
    """Images in `screening-v2` and not in `screening-20k`: what a run must still embed."""
    delta_publishable: int = 0


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


def same_shoot(a: Record, b: Record) -> bool:
    """Two Mémoire photos from one shoot: same photographer and date, or nearby refs.

    Refs are numbered as the prints were filed, so near refs in one series are mostly one
    visit (see `SHOOT_REF_WINDOW`). The ref test also covers photos with no photographer or
    date (0.5% of building photos).
    """
    if a.ref == b.ref:
        return True
    if (
        a.photo_date
        and a.photo_date == b.photo_date
        and set(a.photographers) & set(b.photographers)
    ):
        return True
    ref_a, ref_b = _REF_NUMBER.fullmatch(a.ref), _REF_NUMBER.fullmatch(b.ref)
    if ref_a is None or ref_b is None:
        return False
    same_series = (ref_a[1], ref_a[3]) == (ref_b[1], ref_b[3])
    return same_series and abs(int(ref_a[2]) - int(ref_b[2])) <= SHOOT_REF_WINDOW


def t1_queries(records: Sequence[Record], size: int, seed: int = SEED) -> list[Query]:
    """Mémoire buildings (PA) with 2+ photos. One photo is the query; the others are the answer.

    Photos from the query's shoot leave the gallery (`same_shoot`): they are often the next
    frame on the same film, so finding them tests near-duplicate search, not knowing the
    building from another visit. A query left with no answer stays in the draw, so the halves
    and galleries do not move, and `drop_unanswerable` removes it after the split.
    """
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
        shoot = {r.ref: r for _, r in photos[entity] if same_shoot(record, r)}.values()
        queries.append(
            Query(
                query_id=f"T1-{image_id}",
                set_id="T1",
                entity_id=entity,
                query_image_id=image_id,
                expected_entity_ids=(entity,),
                # The query's photo record and every other photo of its shoot leave the gallery.
                exclude_image_ids=tuple(sorted(i.image_id for r in shoot for i in r.images)),
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
    return split_following(set_id, queries, {}, seed)


def split_following(
    set_id: str,
    queries: Iterable[Query],
    fixed: Mapping[str, Half],
    seed: int = SEED,
    draw: str | None = None,
) -> SplitSet:
    """Like `split_by_entity`, but entities already in a half of another set keep it.

    The other entities are shuffled and fill the halves so that, together, half the entities
    (rounded up) are in `tune`. `draw` names the shuffle when two draws share a set id.
    """
    queries = sorted(queries, key=lambda q: q.query_id)
    entities = sorted({q.entity_id for q in queries})
    free = [e for e in entities if e not in fixed]
    _rng(seed, f"{draw or set_id}:split").shuffle(free)
    tune_fixed = {e for e in entities if fixed.get(e) == "tune"}
    room = min(max((len(entities) + 1) // 2 - len(tune_fixed), 0), len(free))
    tune_entities = tune_fixed | set(free[:room])
    return SplitSet(
        set_id,
        tune=tuple(q for q in queries if q.entity_id in tune_entities),
        sealed=tuple(q for q in queries if q.entity_id not in tune_entities),
    )


def halves_of(splits: Iterable[SplitSet]) -> dict[str, Half]:
    """The half each entity sits in."""
    halves: dict[str, Half] = {}
    for split in splits:
        halves.update(dict.fromkeys((q.entity_id for q in split.tune), "tune"))
        halves.update(dict.fromkeys((q.entity_id for q in split.sealed), "sealed"))
    return halves


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
    """A closed model may run the query: its image is publishable, and so is one answer image.

    Commons photos are openly licensed, so their own image always may go.
    """
    commons = is_commons_image(query.query_image_id)
    if not commons and not index.image_publishable[query.query_image_id]:
        return False
    answers = (i for e in query.expected_entity_ids for i in index.entity_images[e])
    return not query.expected_entity_ids or any(i in publishable_gallery for i in answers)


def commons_queries(
    photos: Iterable[CommonsPhoto], index: RecordIndex
) -> tuple[dict[CommonsSet, list[Query]], int]:
    """One query per Commons photo, and how many were dropped because the table changed.

    T2 and T2b need an image of the entity in the table; a T8 photo needs none. The Commons
    photo is never in a gallery, so there is nothing to exclude.
    """
    queries: dict[CommonsSet, list[Query]] = {s: [] for s in CommonsSet}
    dropped = 0
    for p in photos:
        in_table = p.entity_id in index.entity_images
        if in_table == (p.set_id == CommonsSet.T8):
            dropped += 1
            continue
        expected = () if p.set_id == CommonsSet.T8 else (p.entity_id,)
        queries[p.set_id].append(
            Query(
                query_id=f"{p.set_id.value}-{p.image_id}",
                set_id=p.set_id.value,
                entity_id=p.entity_id,
                query_image_id=p.image_id,
                expected_entity_ids=expected,
                exclude_image_ids=(),
            )
        )
    if dropped:
        logger.warning("%d Commons photos no longer fit their set; rebuild them", dropped)
    return queries, dropped


def answer_images(queries: Iterable[Query], index: RecordIndex) -> set[str]:
    """Every image of every right answer: what a gallery needs for these queries."""
    return {i for q in queries for e in q.expected_entity_ids for i in index.entity_images[e]}


# --- the whole build ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuiltSets:
    splits: tuple[SplitSet, ...]
    """What gets written: queries with no answer left are gone."""
    drawn: tuple[SplitSet, ...]
    """Every POP query drawn, before `drop_unanswerable`. Galleries are built from these, so a
    stricter answer rule does not move the screening gallery or the runs embedded on it."""
    dropped: dict[str, int]
    commons: tuple[SplitSet, ...] = ()
    """T2, T2b and the Commons part of T8 (also inside T8 in `splits`)."""


def answerable(query: Query, index: RecordIndex) -> bool:
    """Some image of a right entity is left once the query's exclusions leave the gallery."""
    hidden = {query.query_image_id, *query.exclude_image_ids}
    return any(i not in hidden for e in query.expected_entity_ids for i in index.entity_images[e])


def drop_unanswerable(split: SplitSet, index: RecordIndex) -> SplitSet:
    """Remove queries whose every answer is excluded. The halves stay as drawn."""
    kept = SplitSet(
        split.set_id,
        tune=tuple(q for q in split.tune if answerable(q, index)),
        sealed=tuple(q for q in split.sealed if answerable(q, index)),
    )
    dropped = len(split.queries) - len(kept.queries)
    if dropped:
        logger.info("%s: dropped %d queries with no answer left", split.set_id, dropped)
    return kept


def build_sets(
    records: Sequence[Record],
    sizes: Sizes,
    seed: int = SEED,
    commons: Sequence[CommonsPhoto] = (),
) -> BuiltSets:
    index = RecordIndex(records)
    t1 = split_by_entity("T1", t1_queries(records, sizes.t1, seed), seed)
    t3 = split_by_entity("T3", t3_queries(records, seed), seed)
    t5 = split_by_entity("T5", t5_queries(records, sizes.t5, seed), seed)
    # T8 needs no answer, so it draws from every T1 and T5 query, answerable or not.
    t8 = t8_split((t1, t5), index, sizes.t8_per_part, seed)
    drawn = (t1, t3, t5, t8)
    splits = [s if s.set_id == "T8" else drop_unanswerable(s, index) for s in drawn]
    dropped = {
        s.set_id: len(d.queries) - len(s.queries) for s, d in zip(splits, drawn, strict=True)
    }
    commons_splits: tuple[SplitSet, ...] = ()
    if commons:
        by_set, commons_dropped = commons_queries(commons, index)
        # Entities already in a half keep it, so no entity is in both halves across sets.
        fixed = halves_of(drawn)
        parts: list[SplitSet] = []
        for commons_set in CommonsSet:
            label = commons_set.value
            split = split_following(
                label, by_set[commons_set], fixed, seed, draw=f"{label}-commons"
            )
            fixed.update(halves_of([split]))
            parts.append(split)
        commons_splits = tuple(parts)
        t2, t2b, t8_commons = commons_splits
        merged_t8 = SplitSet("T8", t8.tune + t8_commons.tune, t8.sealed + t8_commons.sealed)
        splits = [merged_t8 if s.set_id == "T8" else s for s in splits]
        splits.extend((t2, t2b))
        dropped.update({"T2": 0, "T2b": 0})
        dropped["T8"] += commons_dropped
    check_halves_apart((*drawn, *commons_splits))
    return BuiltSets(splits=tuple(splits), drawn=drawn, dropped=dropped, commons=commons_splits)


def build(settings: Settings, sizes: Sizes | None = None, seed: int = SEED) -> BuildReport:
    sizes = sizes or Sizes()
    records = read_records(settings.records_parquet, settings.images_parquet)
    commons = read_commons(settings.commons_parquet) if settings.commons_parquet.exists() else []
    if not commons:
        logger.info("no Commons photos (run `miroir testsets commons`): T2 and T2b skipped")
    index = RecordIndex(records)
    built = build_sets(records, sizes, seed, commons)
    for split in built.splits:
        write_set(settings, split.set_id, split.tune, split.sealed)

    all_ids = set(index.image_publishable)
    needed = needed_images((q for s in built.drawn for q in s.queries), index)
    screening = screening_gallery(all_ids, needed, sizes.screening, seed)
    publishable = {i for i in screening if index.image_publishable[i]}
    write_gallery(settings, FULL, all_ids)
    write_gallery(settings, SCREENING, screening)
    write_gallery(settings, SCREENING_PUBLISHABLE, publishable)

    v2 = screening | answer_images((q for s in built.commons for q in s.queries), index)
    v2_publishable = {i for i in v2 if index.image_publishable[i]}
    if built.commons:
        write_gallery(settings, SCREENING_V2, v2)
        write_gallery(settings, SCREENING_V2_PUBLISHABLE, v2_publishable)
        write_gallery(settings, SCREENING_V2 + DELTA, v2 - screening)
        write_gallery(settings, SCREENING_V2_PUBLISHABLE + DELTA, v2_publishable - publishable)

    reports = tuple(
        _set_report(s, built, index, v2_publishable if s.set_id in MIN_SIZES else publishable)
        for s in built.splits
    )
    for r in reports:
        logger.info(
            "%s: %d tune, %d sealed, %d dropped (spec %s%d)",
            r.set_id,
            r.tune,
            r.sealed,
            r.dropped,
            "at least " if r.at_least else "",
            r.spec,
        )
        if not r.meets_spec:
            logger.warning(
                "%s has %d queries; spec §7 says %d", r.set_id, r.tune + r.sealed, r.spec
            )
    return BuildReport(
        sets=reports,
        full=len(all_ids),
        screening=len(screening),
        needed=len(needed),
        screening_publishable=len(publishable),
        screening_v2=len(v2) if built.commons else 0,
        screening_v2_publishable=len(v2_publishable) if built.commons else 0,
        delta=len(v2 - screening) if built.commons else 0,
        delta_publishable=len(v2_publishable - publishable) if built.commons else 0,
    )


def _set_report(
    split: SplitSet, built: BuiltSets, index: RecordIndex, publishable: set[str]
) -> SetReport:
    at_least = split.set_id in MIN_SIZES
    if at_least:
        spec = MIN_SIZES[split.set_id]
    elif split.set_id == "T8" and built.commons:
        spec = SPEC_SIZES["T8"] + T8_COMMONS
    else:
        spec = SPEC_SIZES[split.set_id]
    return SetReport(
        set_id=split.set_id,
        tune=len(split.tune),
        sealed=len(split.sealed),
        spec=spec,
        closed_ok=sum(closed_ok(q, publishable, index) for q in split.queries),
        dropped=built.dropped[split.set_id],
        at_least=at_least,
    )
