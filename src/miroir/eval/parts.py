"""Test sets for finding the whole from a part (docs/search-levels.md, "Test sets" 1 and 2).

VET: three photos from outside POP of the Vétheuil Passion altarpiece (PM95000723): two sale
catalogue pages, each one stolen carved group, and the altarpiece today. The target is
APMH047981, a black-and-white Mémoire photo from before 1896; any record of the altarpiece
counts as right. Three photos of one object cannot be split, so VET has a tune half only.

PARTS: pairs mined from the whole POP download (`full-pop/`). The query is a Mémoire photo
captioned as a detail ("..., détail", "..., vue partielle"); the answer is its building or
object, reached through a photo whose caption names the same thing without "détail". Every
detail photo of the entity leaves the gallery. One query per entity, a fixed number per kind,
publishable records only. Halves are split by entity, following the halves of the other sets.
"""

from __future__ import annotations

import logging
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from miroir.config import SEED, Half, Settings
from miroir.eval.build_sets import halves_of, split_following
from miroir.eval.sets import Query, load_queries, write_set
from miroir.eval.words import other_halves
from miroir.records.model import Collection, Image, Record
from miroir.records.store import read_column, read_records

logger = logging.getLogger(__name__)

# --- VET ---------------------------------------------------------------------------------------

VET = "VET"
VET_ENTITY = "PM95000723"
VET_TARGET_REF = "APMH047981"
#: Prefix of query images that are in no POP record (`Settings.outside_image_path`).
OUTSIDE_PREFIX = "outside-"


@dataclass(frozen=True, slots=True)
class OutsideCase:
    number: int
    what: str


#: The three VET photos, in the order `miroir testsets vet` takes their files.
VET_CASES = (
    OutsideCase(1, "sale catalogue page: Mise au tombeau, a stolen carved group, and text"),
    OutsideCase(2, "the altarpiece today, in colour, the stolen groups replaced by copies"),
    OutsideCase(3, "sale catalogue page: Descente de croix, a stolen carved group, and text"),
)


def vet_queries() -> list[Query]:
    """The three VET queries, one per photo; each expects the altarpiece."""
    return [
        Query(
            query_id=f"{VET}-{case.number}",
            set_id=VET,
            entity_id=VET_ENTITY,
            query_image_id=f"{OUTSIDE_PREFIX}{VET}-{case.number}",
            expected_entity_ids=(VET_ENTITY,),
            exclude_image_ids=(),
            kind=Kind.ALTARPIECE.value,
            why=f"{case.what}; target {VET_TARGET_REF}, Mémoire photo from before 1896",
        )
        for case in VET_CASES
    ]


def build_vet(settings: Settings, images: Sequence[Path]) -> list[Query]:
    """Copy the three photos into the test sets' outside folder and write VET (tune only)."""
    if len(images) != len(VET_CASES):
        msg = f"VET takes {len(VET_CASES)} images, in the order of VET_CASES; got {len(images)}"
        raise ValueError(msg)
    queries = vet_queries()
    for query, source in zip(queries, images, strict=True):
        target = settings.outside_image_path(query.query_image_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_set(settings, VET, queries, [])
    return queries


# --- PARTS: reading captions ------------------------------------------------------------------

PARTS = "PARTS"


class Kind(StrEnum):
    ALTARPIECE = "altarpiece"
    STATUE = "statue"
    PAINTING = "painting"
    FURNITURE = "furniture"
    STAINED_GLASS = "stained-glass"
    BUILDING = "building"


#: Pairs drawn per kind.
PER_KIND = 60

#: Palissy denominations (lower case, "(2)" dropped) of each object kind. A retable anywhere in
#: the list wins: "autel, retable, tabernacle" is an altarpiece.
_DENOMINATIONS: dict[Kind, frozenset[str]] = {
    Kind.ALTARPIECE: frozenset({"retable", "retables", "autel-retable", "retable architecturé"}),
    Kind.STATUE: frozenset(
        {
            "statue",
            "statues",
            "statuette",
            "groupe sculpté",
            "buste",
            "bas-relief",
            "haut-relief",
            "relief",
            "sculpture",
            "crucifix",
            "christ en croix",
        }
    ),
    Kind.PAINTING: frozenset({"tableau", "tableaux", "peinture", "peinture monumentale"}),
    Kind.FURNITURE: frozenset(
        {
            "stalle",
            "stalles",
            "chaire à prêcher",
            "confessionnal",
            "lambris de revêtement",
            "lambris",
            "armoire",
            "buffet",
            "buffet d'orgue",
            "banc",
            "banc d'oeuvre",
            "lutrin",
            "clôture de choeur",
            "fauteuil",
            "commode",
            "coffre",
            "table",
            "meuble",
            "porte",
            "vantail",
            "tabernacle",
        }
    ),
    Kind.STAINED_GLASS: frozenset({"verrière", "verrières", "vitrail", "vitraux"}),
}

#: A building detail must end on the word: "façade sud, détail".
_BUILDING_DETAIL = re.compile(r"[,:]\s*d[ée]tails?\s*\.?$")
_DETAIL = re.compile(r"\b(?:vue de )?d[ée]tails?\b|\bvue partielle\b|\bgros plan\b")
#: Details nobody can match by eye to the whole: marks, writing, backs, cloth, damage.
_BLIND = re.compile(
    r"poin[çc]on|inscri|signature|sign[ée]|\bmarques?\b|monogramme|cachet|estampille|"
    r"[ée]tiquette|revers|\bdos\b|envers|verso|\bdates?\b|dat[ée]|graffit|\bchiffres?\b|"
    r"\btexte|l[ée]gende|cartouche|\blettres?\b|[ée]pitaphe|mill[ée]sime|serrure|ferrure|"
    r"charni[èe]re|fixation|\b[ée]tat\b|d[ée]gradation|restauration|fissure|tissu|broderie|"
    r"galon|frange|dentelle|orfroi|motif|ornement|\bfond\b"
)
#: Words that say a non-detail photo shows the whole thing.
_WHOLE = re.compile(r"vue g[ée]n[ée]rale|vue d'ensemble|\bensemble\b|vue de face")
_SEPARATORS = " \t.,;:()-\u2013\u2014'\""


def caption(record: Record) -> str:
    """A Mémoire photo's caption, lower case and tidy: its description, else its title."""
    return _tidy(record.description or record.title)


def _tidy(text: str) -> str:
    """Lower case, one kind of apostrophe, single spaces."""
    text = unicodedata.normalize("NFC", text).replace("’", "'").lower()
    return " ".join(text.split())


#: Captions that open on an object: "tableau : ...", "statue de ...", "verrière, détail".
_OBJECT = re.compile(
    "^(?:"
    + "|".join(re.escape(n) for names in _DENOMINATIONS.values() for n in sorted(names))
    + r"|autel|tapisserie|pièce murale|cloche|croix|orgue|lustre|objet|mobilier|dalle|plaque"
    + r"|tombeau|gisant|bannière|calice|reliquaire|tenture|vierge|christ|saint|sainte)\b"
)


def is_object(text: str) -> bool:
    """The caption names an object, not a part of a building."""
    return _OBJECT.match(text) is not None


def is_detail(text: str) -> bool:
    return _DETAIL.search(text) is not None


def is_blind(text: str) -> bool:
    return _BLIND.search(text) is not None


def detail_subject(text: str) -> str:
    """What a detail is of, when the caption says so first: "verrière de la vierge, détail :
    saint jean" -> "verrière de la vierge". Empty when the caption opens on "détail"."""
    match = _DETAIL.search(text)
    return "" if match is None else text[: match.start()].strip(_SEPARATORS)


def whole_subject(text: str) -> str:
    """What a whole view shows: "verrière de la vierge, vue générale" -> "verrière de la vierge"."""
    return _WHOLE.sub("", text).strip(_SEPARATORS).replace(" ,", ",")


def _denomination_kind(denominations: Iterable[str]) -> Kind | None:
    names = [re.sub(r"\s*\(\d+\)$", "", d.strip().lower()) for d in denominations]
    if any("retable" in n for n in names):
        return Kind.ALTARPIECE
    for name in names:
        for kind, known in _DENOMINATIONS.items():
            if name in known:
                return kind
    return None


# --- PARTS: pairs ------------------------------------------------------------------------------


class Rule(StrEnum):
    """Why a photo counts as the whole of a detail, strongest first.

    Looser rules were tried and dropped after a check by eye of 30 pairs: a photo captioned
    "vue générale" of the same object, and the object's own Palissy record image, mostly showed
    another part of it; Rose Valland images typed `detail` include backs and stamps.
    """

    SAME_SUBJECT = "same subject"
    """The whole's caption names what the detail's caption says it is a detail of."""
    CONTAINS = "contains"
    """Objects only: the detail's subject starts with the whole's caption ("retable" holds
    "retable, tableau : saint maurice")."""


@dataclass(frozen=True, slots=True)
class Pair:
    query: Query
    rule: Rule


def entity_pair(
    entity_id: str, records: Sequence[Record], kind: Kind, rng: random.Random
) -> Pair | None:
    """One detail photo of the entity and the photo of its whole that answers it, or None.

    A building's query must pass `_building_part`: a "vue partielle" of a building, or a detail
    of something named after "détail", was mostly another view in a check by eye.
    """
    building = kind == Kind.BUILDING
    photos = [r for r in records if r.collection == Collection.MEMOIRE and r.images]
    details = [r for r in photos if is_detail(caption(r))]
    queries = [r for r in details if r.publishable and not is_blind(caption(r))]
    if building:
        queries = [r for r in queries if _building_part(r)]
    wholes = [r for r in photos if not is_detail(caption(r)) and not is_blind(caption(r))]
    hidden = sorted(i.image_id for r in details for i in r.images)
    for query in _shuffled(queries, rng):
        found = _whole_for(query, wholes, building=building)
        if found is None:
            continue
        target, rule = found
        query_image, target_image = query.images[0], target.images[0]
        if query_image.url == target_image.url:
            continue
        why = f'{rule.value}: detail "{caption(query)}"; whole {target.ref} "{caption(target)}"'
        return Pair(_query(entity_id, query_image, hidden, kind, why), rule)
    return None


def _building_part(record: Record) -> bool:
    """A building photo that is a close view of one named part: "portail ouest, détail".

    Not "chapelle, détail de l'autel" (the whole view may not show the altar), not the building
    itself ("église saint-jean, détail"), and not an object photographed without its record.
    """
    text = caption(record)
    return (
        _BUILDING_DETAIL.search(text) is not None
        and not is_object(text)
        and detail_subject(text) != _tidy(record.title).strip(_SEPARATORS)
    )


def _whole_for(
    detail: Record, wholes: Sequence[Record], *, building: bool
) -> tuple[Record, Rule] | None:
    """The best whole for a detail: a publishable photo whose caption names its subject."""
    subject = detail_subject(caption(detail))
    if len(subject) < 3:
        return None
    ranked: list[tuple[Rule, str, Record]] = []
    for whole in wholes:
        if not whole.publishable:
            continue
        seen = whole_subject(caption(whole))
        if seen == subject:
            ranked.append((Rule.SAME_SUBJECT, whole.ref, whole))
        elif (
            not building
            and len(seen) >= 6
            and subject.startswith(seen)
            and subject[len(seen)] in _SEPARATORS
        ):
            ranked.append((Rule.CONTAINS, whole.ref, whole))
    order = list(Rule)
    ranked.sort(key=lambda item: (order.index(item[0]), item[1]))
    return (ranked[0][2], ranked[0][0]) if ranked else None


def _query(entity_id: str, image: Image, hidden: list[str], kind: Kind, why: str) -> Query:
    return Query(
        query_id=f"{PARTS}-{image.image_id}",
        set_id=PARTS,
        entity_id=entity_id,
        query_image_id=image.image_id,
        expected_entity_ids=(entity_id,),
        exclude_image_ids=tuple(hidden),
        kind=kind.value,
        why=why,
    )


def _shuffled[T](items: Sequence[T], rng: random.Random) -> list[T]:
    out = list(items)
    rng.shuffle(out)
    return out


# --- PARTS: the draw ---------------------------------------------------------------------------


def candidate_kinds(records_path: Path) -> dict[str, Kind]:
    """Every entity that may give a pair, with its kind, from a few columns of the full table.

    Candidates: Mémoire buildings and objects with a publishable photo captioned as a detail.
    An object's kind comes from its Palissy record; one without that record is left out.
    """
    entity = [str(v) for v in read_column(records_path, "entity_id")]
    base = [str(v) for v in read_column(records_path, "collection")]
    text = [str(v).lower() for v in read_column(records_path, "description")]
    publishable = [bool(v) for v in read_column(records_path, "publishable")]
    denomination = read_column(records_path, "denomination")
    memoire, palissy = Collection.MEMOIRE.value, Collection.PALISSY.value
    wanted = {
        e
        for e, b, t, p in zip(entity, base, text, publishable, strict=True)
        if b == memoire and p and e.startswith(("PA", "PM")) and is_detail(t)
    }
    kinds: dict[str, Kind] = {e: Kind.BUILDING for e in wanted if e.startswith("PA")}
    for e, b, names in zip(entity, base, denomination, strict=True):
        if b == palissy and e in wanted:
            kind = _denomination_kind(_strings(names))
            if kind is not None:
                kinds[e] = kind
    kinds.pop(VET_ENTITY, None)  # the VET set's own altarpiece stays out of PARTS
    logger.info(
        "PARTS: %d candidate entities, %d with a kind: %s",
        len(wanted),
        len(kinds),
        dict(Counter(k.value for k in kinds.values())),
    )
    return kinds


def _strings(value: object) -> list[str]:
    return [str(v) for v in cast("list[object]", value)] if isinstance(value, list) else []


@dataclass(frozen=True, slots=True)
class PartsReport:
    """What `build_parts` wrote. Counters are keyed by kind, except `rules`."""

    tune: Counter[str]
    """Queries per kind in the tune half."""
    sealed: Counter[str]
    """Queries per kind in the sealed half."""
    rules: Counter[str]
    """Pairs per `Rule` that found their whole."""
    tried: Counter[str]
    """Entities read per kind before the kind had its pairs."""


def draw_pairs(
    settings: Settings, kinds: Mapping[str, Kind], per_kind: int = PER_KIND, seed: int = SEED
) -> tuple[list[Pair], Counter[str]]:
    """Up to `per_kind` pairs per kind, reading candidates in a seeded order, a batch at a time."""
    rng = random.Random(f"{seed}:{PARTS}")
    queues: dict[Kind, list[str]] = {}
    for kind in Kind:
        queue = sorted(e for e, k in kinds.items() if k == kind)
        rng.shuffle(queue)
        queues[kind] = queue
    pairs: dict[Kind, list[Pair]] = defaultdict(list)
    tried: Counter[str] = Counter()
    while True:
        batch: dict[str, Kind] = {}
        for kind, queue in queues.items():
            need = per_kind - len(pairs[kind])
            if need > 0 and queue:
                take = queue[: need * 3]
                del queue[: need * 3]
                batch.update(dict.fromkeys(take, kind))
        if not batch:
            break
        by_entity: defaultdict[str, list[Record]] = defaultdict(list)
        for record in read_records(
            settings.full_records_parquet, settings.full_images_parquet, batch
        ):
            by_entity[record.entity_id].append(record)
        for entity in sorted(batch, key=lambda e: (list(Kind).index(batch[e]), e)):
            kind = batch[entity]
            if len(pairs[kind]) >= per_kind:
                continue
            tried[kind.value] += 1
            records = sorted(by_entity[entity], key=lambda r: r.ref)
            pair = entity_pair(entity, records, kind, random.Random(f"{seed}:{PARTS}:{entity}"))
            if pair is not None:
                pairs[kind].append(pair)
        logger.info("PARTS: %s", {k.value: len(v) for k, v in pairs.items()})
    for kind in Kind:
        if len(pairs[kind]) < per_kind:
            logger.warning("PARTS %s: only %d pairs, wanted %d", kind, len(pairs[kind]), per_kind)
    return [p for kind in Kind for p in pairs[kind]], tried


def fixed_halves(settings: Settings) -> dict[str, Half]:
    """Halves the PARTS entities must follow: every other set's, and VET's altarpiece in tune."""
    records = read_records(settings.records_parquet, settings.images_parquet)
    fixed = other_halves(settings, records)
    for set_id in ("T7", "T7en"):  # only their tune half may be read; its entities stay there
        if settings.testset_path(set_id, "tune").exists():
            fixed.update(
                dict.fromkeys((q.entity_id for q in load_queries(settings, set_id, "tune")), "tune")
            )
    fixed[VET_ENTITY] = "tune"
    return fixed


def build_parts(settings: Settings, per_kind: int = PER_KIND, seed: int = SEED) -> PartsReport:
    """Draw the pairs from the full POP tables, split them into halves and write PARTS."""
    kinds = candidate_kinds(settings.full_records_parquet)
    pairs, tried = draw_pairs(settings, kinds, per_kind, seed)
    fixed = fixed_halves(settings)
    # Each kind is split on its own, so both halves hold every kind about equally.
    splits = [
        split_following(
            f"{PARTS}-{kind.value}", [p.query for p in pairs if p.query.kind == kind], fixed, seed
        )
        for kind in Kind
    ]
    tune = [q for s in splits for q in s.tune]
    sealed = [q for s in splits for q in s.sealed]
    write_set(settings, PARTS, tune, sealed)
    halves = halves_of(splits)
    logger.info(
        "PARTS: %d entities, %d tune", len(halves), sum(h == "tune" for h in halves.values())
    )
    return PartsReport(
        tune=Counter(q.kind for q in tune),
        sealed=Counter(q.kind for q in sealed),
        rules=Counter(p.rule.value for p in pairs),
        tried=tried,
    )
