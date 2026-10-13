"""Test set T7: the words a member of the public might type for a hidden image (§7, F2).

A vision model (`WRITER`, not from the Google family, since EmbeddingGemma 2 is scored) sees
one image and writes the search a visitor standing in front of the building or object would
type. The image then leaves the index, so the words must find the record through its other
images or its text. The answer is the image's building or object.

Draw: 400 Mémoire buildings (PA), 400 Joconde objects and 200 Rose Valland works (no Jeu de
Paume crops), each with an answer left once the hidden images are out: for a building, every
photo of the query's shoot (T1's rule) leaves; for an object, the query image alone. Only
publishable `main` images in the website's index are drawn, since they go to OpenRouter.
Halves split by entity; entities already in a half of another set keep it.

Each image gets a seeded length (a few words, one sentence, two sentences) and register
(plain or expert words), so lengths and vocabulary vary. A seeded `ENGLISH` of them also get an
English version of the same search: set T7en, to compare languages on the same images.
"""

from __future__ import annotations

import logging
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from pydantic import TypeAdapter, ValidationError

from miroir.config import SEED, Half, Settings
from miroir.eval.build_sets import (
    JEU_DE_PAUME_PREFIX,
    Sizes,
    SplitSet,
    build_sets,
    halves_of,
    same_shoot,
    split_following,
)
from miroir.eval.sets import Query, load_queries, write_set
from miroir.jobs.describe import DescribeOption, read_answers
from miroir.records.model import Collection, ImageType, Record
from miroir.records.store import read_commons, read_records

logger = logging.getLogger(__name__)

SET_ID = "T7"
EN_SET_ID = "T7en"
SIZES: dict[Collection, int] = {
    Collection.MEMOIRE: 400,
    Collection.JOCONDE: 400,
    Collection.ROSE_VALLAND: 200,
}
#: Images that also get an English version (about half in each half).
ENGLISH = 200
#: The writer. Qwen3-VL has no `:batch` variant on OpenRouter (checked 2026-10-13), so the
#: standard endpoint. A warm temperature varies the wording from one image to the next.
WRITER = DescribeOption(
    "T7-qwen3-vl-30b-a3b",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "t7-qwen3-vl-30b-a3b",
    batch=False,
    extra={"temperature": 0.8, "response_format": {"type": "json_object"}},
)


class Length(StrEnum):
    WORDS = "words"
    SENTENCE = "sentence"
    LONG = "long"


class Register(StrEnum):
    PLAIN = "plain"
    EXPERT = "expert"


LENGTH_SHARES = ((Length.WORDS, 0.45), (Length.SENTENCE, 0.35), (Length.LONG, 0.20))
REGISTER_SHARES = ((Register.PLAIN, 0.7), (Register.EXPERT, 0.3))

_LENGTH_RULE = {
    Length.WORDS: "Écris 3 à 6 mots, comme dans un moteur de recherche, sans phrase.",
    Length.SENTENCE: "Écris une seule phrase courte (8 à 15 mots).",
    Length.LONG: "Écris deux phrases (20 à 35 mots en tout) avec les détails qui frappent.",
}
_REGISTER_RULE = {
    Register.PLAIN: "Emploie les mots de tous les jours d'un visiteur qui n'est pas spécialiste.",
    Register.EXPERT: (
        "Emploie le vocabulaire précis d'un amateur éclairé (termes d'architecture, "
        "de technique ou de style)."
    ),
}
_PROMPT = """Voici la photo d'un édifice, d'un objet ou d'une œuvre du patrimoine français. \
Un visiteur a cette chose devant les yeux et la cherche dans une base du patrimoine en tapant \
ce qu'il voit. Écris, en français, la recherche qu'il taperait.
- {length}
- {register}
- Décris la chose : type d'édifice ou d'objet, forme, matière, couleurs, sujet représenté, \
détails marquants.
- Ne parle pas de la photo elle-même (noir et blanc, tirage, cadrage, fond) : le visiteur \
voit la chose, pas cette photo.
- N'utilise rien de ce qui est écrit sur la photo, son carton ou une étiquette (légende, \
date, nom de lieu ou de personne, cote, numéro) : le visiteur ne le voit pas.
- Ne donne un nom propre (lieu, artiste, titre) que pour un monument ou une œuvre très \
célèbre que tout le monde reconnaîtrait ; sinon n'en invente pas.
Réponds uniquement par un objet JSON : {keys}"""
_KEYS_FR = '{"fr": "<la recherche>"}'
_KEYS_EN = (
    '{"fr": "<la recherche>", "en": "<la même recherche, telle qu\'un visiteur anglophone '
    'la taperait en anglais>"}'
)


@dataclass(frozen=True, slots=True)
class Draw:
    """One T7 image before its words are written: its query, half and writing brief."""

    query: Query
    half: Half
    collection: Collection
    length: Length
    register: Register
    english: bool

    @property
    def prompt(self) -> str:
        return _PROMPT.format(
            length=_LENGTH_RULE[self.length],
            register=_REGISTER_RULE[self.register],
            keys=_KEYS_EN if self.english else _KEYS_FR,
        )


# --- the draw --------------------------------------------------------------------------------


def draw(
    records: Sequence[Record],
    index_ids: set[str],
    fixed: Mapping[str, Half],
    seed: int = SEED,
    sizes: Mapping[Collection, int] = SIZES,
) -> tuple[Draw, ...]:
    """Every T7 image, both halves, in query id order. Same records, same draw."""
    queries = [
        *_buildings(records, index_ids, sizes[Collection.MEMOIRE], seed),
        *_objects(records, index_ids, Collection.JOCONDE, sizes[Collection.JOCONDE], seed),
        *_objects(
            records, index_ids, Collection.ROSE_VALLAND, sizes[Collection.ROSE_VALLAND], seed
        ),
    ]
    split = split_following(SET_ID, queries, fixed, seed)
    half_of: dict[str, Half] = {q.query_id: "tune" for q in split.tune}
    half_of.update({q.query_id: "sealed" for q in split.sealed})
    collection_of = {r.entity_id: r.collection for r in records}
    ordered = sorted(queries, key=lambda q: q.query_id)
    rng = random.Random(f"{seed}:{SET_ID}:brief")
    english = set(rng.sample([q.query_id for q in ordered], min(ENGLISH, len(ordered))))
    out: list[Draw] = []
    for q in ordered:
        out.append(
            Draw(
                query=q,
                half=half_of[q.query_id],
                collection=collection_of[q.entity_id],
                length=_pick(rng, LENGTH_SHARES),
                register=_pick(rng, REGISTER_SHARES),
                english=q.query_id in english,
            )
        )
    logger.info(
        "T7 draw: %d images (%s), %d tune, %d with English",
        len(out),
        dict(Counter(d.collection.value for d in out)),
        sum(d.half == "tune" for d in out),
        len(english),
    )
    return tuple(out)


def _pick[T](rng: random.Random, shares: Sequence[tuple[T, float]]) -> T:
    return rng.choices([v for v, _ in shares], weights=[w for _, w in shares])[0]


def _main_images(record: Record, index_ids: set[str]) -> list[str]:
    if not record.publishable:
        return []
    return sorted(
        i.image_id
        for i in record.images
        if i.image_type is ImageType.MAIN and i.image_id in index_ids
    )


def _buildings(records: Sequence[Record], index_ids: set[str], size: int, seed: int) -> list[Query]:
    """Mémoire buildings: one photo, hidden with every photo of its shoot (T1's rule)."""
    photos: defaultdict[str, list[Record]] = defaultdict(list)
    for r in records:
        if r.collection is Collection.MEMOIRE and r.entity_id.startswith("PA"):
            photos[r.entity_id].append(r)
    entities = sorted(photos)
    rng = random.Random(f"{seed}:{SET_ID}:memoire")
    rng.shuffle(entities)
    out: list[Query] = []
    for entity in entities:
        if len(out) == size:
            break
        candidates = [(i, r) for r in photos[entity] for i in _main_images(r, index_ids)]
        if not candidates:
            continue
        image_id, record = rng.choice(sorted(candidates, key=lambda c: c[0]))
        hidden = {i.image_id for r in photos[entity] if same_shoot(record, r) for i in r.images}
        left = [
            i.image_id
            for r in photos[entity]
            for i in r.images
            if i.image_id in index_ids and i.image_id not in hidden
        ]
        if left:
            out.append(_query(image_id, entity, hidden | {image_id}))
    _warn_short(Collection.MEMOIRE, len(out), size)
    return out


def _objects(
    records: Sequence[Record], index_ids: set[str], collection: Collection, size: int, seed: int
) -> list[Query]:
    """Joconde or Rose Valland records with 2+ images in the index; the query image hides."""
    eligible = sorted(
        (
            r
            for r in records
            if r.collection is collection
            and not r.ref.startswith(JEU_DE_PAUME_PREFIX)
            and sum(i.image_id in index_ids for i in r.images) >= 2
            and _main_images(r, index_ids)
        ),
        key=lambda r: r.ref,
    )
    rng = random.Random(f"{seed}:{SET_ID}:{collection.value}")
    chosen = rng.sample(eligible, min(size, len(eligible)))
    _warn_short(collection, len(chosen), size)
    return [
        _query(image_id, r.entity_id, {image_id})
        for r in chosen
        for image_id in [rng.choice(_main_images(r, index_ids))]
    ]


def _query(image_id: str, entity: str, hidden: set[str]) -> Query:
    return Query(
        query_id=f"{SET_ID}-{image_id}",
        set_id=SET_ID,
        entity_id=entity,
        query_image_id=image_id,
        expected_entity_ids=(entity,),
        exclude_image_ids=tuple(sorted(hidden)),
    )


def _warn_short(collection: Collection, found: int, size: int) -> None:
    if found < size:
        logger.warning("T7 %s: only %d eligible, wanted %d", collection.value, found, size)


def other_halves(settings: Settings, records: Sequence[Record]) -> dict[str, Half]:
    """The half of every entity in another set, from the same in-memory draw `testsets build`
    makes (no sealed file is read). Fails if that draw no longer matches the tune files."""
    commons = read_commons(settings.commons_parquet) if settings.commons_parquet.exists() else []
    built = build_sets(records, Sizes(), SEED, commons)
    for split in built.splits:
        on_disk = {q.query_id for q in load_queries(settings, split.set_id, "tune")}
        if on_disk != {q.query_id for q in split.tune}:
            msg = f"{split.set_id}: the tune file differs from a fresh draw; rebuild the sets"
            raise RuntimeError(msg)
    splits: tuple[SplitSet, ...] = (*built.drawn, *built.commons)
    return halves_of(splits)


def draw_from_settings(settings: Settings, index_ids: set[str]) -> tuple[Draw, ...]:
    records = read_records(settings.records_parquet, settings.images_parquet)
    return draw(records, index_ids, other_halves(settings, records))


# --- the writer's answers --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Written:
    fr: str
    en: str


_OBJECT = TypeAdapter(dict[str, object])
_SPACES = re.compile(r"\s+")


def parse_written(text: str) -> Written | None:
    """The French (and English) search in a writer answer, or None if it holds no French."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        raw = _OBJECT.validate_json(match.group(0))
    except ValidationError:
        return None
    fr, en = _clean(raw.get("fr")), _clean(raw.get("en"))
    return Written(fr, en) if fr else None


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _SPACES.sub(" ", value).strip().strip("\"'«» ").strip()


def build_sets_from_answers(settings: Settings, draws: Sequence[Draw]) -> dict[str, int]:
    """Write T7 and T7en (both halves) from the saved answers. Returns queries per set/half.

    An image with no usable answer is left out and counted.
    """
    answers = read_answers(settings.vlm_answers_path(WRITER.slug))
    halves: dict[tuple[str, Half], list[Query]] = defaultdict(list)
    missing = 0
    for d in draws:
        answer = answers.get(d.query.query_image_id)
        written = None if answer is None else parse_written(answer.text)
        if written is None:
            missing += 1
            continue
        halves[SET_ID, d.half].append(replace(d.query, text=written.fr))
        if d.english and written.en:
            english = replace(
                d.query, query_id=f"{EN_SET_ID}-{d.query.query_image_id}", set_id=EN_SET_ID
            )
            halves[EN_SET_ID, d.half].append(replace(english, text=written.en))
    if missing:
        logger.warning("T7: %d of %d images have no usable answer; left out", missing, len(draws))
    for set_id in (SET_ID, EN_SET_ID):
        write_set(settings, set_id, halves[set_id, "tune"], halves[set_id, "sealed"])
    return {f"{s} {h}": len(qs) for (s, h), qs in sorted(halves.items())}
