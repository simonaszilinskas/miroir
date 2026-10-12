"""Find Commons photos for T2, T2b and T8c; write `commons.parquet`.

Ground truth comes from official ids on Wikidata: Mérimée (P380) for buildings, Joconde (P347)
for museum works. For each item we look at its image (P18) and the newest files of its Commons
category (P373), keep those that look like a visitor's recent photo (`select.photo_problem`)
and pick a few per entity (`select.pick`). Every answer is cached on disk, so a second run
sends no request. Technical spec §7 defines the sets and lists the label noise.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import batched

import httpx

from miroir.commons import api
from miroir.commons.api import CommonsFile, WikidataItem
from miroir.commons.client import WikimediaClient, open_client
from miroir.commons.model import CommonsPhoto, CommonsSet, Source, View, commons_image_id
from miroir.commons.select import (
    Candidate,
    Kind,
    Reject,
    is_view_subcategory,
    names_place,
    photo_problem,
    pick,
    rank,
    view_of,
)
from miroir.config import SEED, Settings
from miroir.images.fetch import ImageCache, ImageFetcher
from miroir.images.phash import distance, phash
from miroir.records.model import Collection, Image, Record
from miroir.records.store import read_records, write_commons

logger = logging.getLogger(__name__)

#: A Commons file this close to an image of its Joconde record (bits of 64) is the same picture:
#: the museum's reproduction, not a visitor's photo.
NEAR_COPY_BITS = 12
#: Subcategories (interior, details) opened per building.
SUBCATEGORIES_PER_BUILDING = 2

#: Called to get an image of our table as a 512 px JPEG, or None when it cannot be fetched.
type PopJpeg = Callable[[Image], bytes | None]


@dataclass(frozen=True, slots=True)
class Targets:
    t2: int = 1_000
    """T2 stops at this many photos (entities are drawn at random until then)."""
    t2b: int | None = None
    """T2b takes every eligible work unless this caps it."""
    per_entity: int = 3
    t8_buildings: int = 200
    t8_artworks: int = 100
    near_copy_bits: int = NEAR_COPY_BITS
    hashed_per_entity: int = 8
    """T2b: candidates checked against the Joconde image, best-ranked first."""


@dataclass(frozen=True, slots=True)
class SetFetch:
    set_id: CommonsSet
    kind: Kind
    photos: int
    entities: int
    """Entities with at least one photo."""
    tried: int
    """Entities looked at."""
    rejects: Mapping[str, int]
    """Commons files left out, by reason."""


@dataclass(frozen=True, slots=True)
class CommonsReport:
    sets: tuple[SetFetch, ...]
    ambiguous: Mapping[str, int]
    """Heritage ids left out because Wikidata ties them to several items, or one item to
    several ids: which building or work a photo shows is then unclear."""
    near_copy_distances: tuple[int, ...]
    """T2b: bits between each checked file and its Joconde image."""
    requests: int
    """Requests sent; 0 when every answer came from the cache."""
    shared: int = 0
    """Files dropped because they were chosen for two entities."""


@dataclass
class _Tally:
    """Counts kept while one set is gathered."""

    rejects: Counter[str] = field(default_factory=Counter[str])
    tried: int = 0
    entities: int = 0


def unambiguous(
    items: Iterable[WikidataItem], keep: Callable[[str], bool]
) -> tuple[dict[str, WikidataItem], int]:
    """Items with exactly one id, held by no other item, and with somewhere to find photos.

    Returns them by id (for ids `keep` accepts), and how many kept ids were left out.
    """
    by_id: defaultdict[str, list[WikidataItem]] = defaultdict(list)
    for item in items:
        for i in item.ids:
            by_id[i].append(item)
    found: dict[str, WikidataItem] = {}
    dropped = 0
    for i, holders in by_id.items():
        if not keep(i):
            continue
        if len(holders) != 1 or len(holders[0].ids) != 1:
            dropped += 1
            continue
        if holders[0].images or holders[0].categories:
            found[i] = holders[0]
    return found, dropped


class CommonsFinder:
    """Gathers, filters and picks Commons photos for each entity."""

    def __init__(
        self,
        client: WikimediaClient,
        pop_jpeg: PopJpeg,
        targets: Targets,
        seed: int = SEED,
    ) -> None:
        self._client = client
        self._pop_jpeg = pop_jpeg
        self._targets = targets
        self._seed = seed
        self._p18: dict[str, CommonsFile] = {}
        self.near_copy_distances: list[int] = []

    def _rng(self, *parts: str) -> random.Random:
        return random.Random(":".join((str(self._seed), "commons", *parts)))

    def load_p18(self, items: Iterable[WikidataItem]) -> None:
        """Image info for the P18 files of `items` not looked up yet, 50 per request."""
        wanted = [t for t in api.titles(list(items)) if t not in self._p18]
        self._p18.update(api.files_by_title(self._client, wanted))

    def candidates(
        self, entity_id: str, item: WikidataItem, kind: Kind, tally: _Tally
    ) -> list[Candidate]:
        """Every usable photo of the item: its P18, its category, and (buildings) view subcats."""
        found: list[tuple[CommonsFile, Source, str, str]] = [
            (self._p18[t], Source.P18, "", "") for t in item.images if t in self._p18
        ]
        for category in item.categories:
            listing = api.category_files(self._client, category)
            found.extend((f, Source.CATEGORY, category, "") for f in listing.files)
            if kind != Kind.BUILDING:
                continue
            views = [s for s in listing.subcategories if is_view_subcategory(s)]
            for sub in views[:SUBCATEGORIES_PER_BUILDING]:
                files = api.category_files(self._client, sub).files
                found.extend((f, Source.SUBCATEGORY, sub, sub) for f in files)
        kept: list[Candidate] = []
        for file, source, category, sub in found:
            problem = photo_problem(file, entity_id, kind)
            if problem is not None:
                tally.rejects[problem] += 1
                continue
            view = view_of(file, sub) if kind == Kind.BUILDING else View.ARTWORK
            named = any(names_place(file, c) for c in item.categories)
            kept.append(Candidate(file, source, category, view, named))
        return rank(kept, self._rng(entity_id))

    def not_copies(
        self,
        entity_id: str,
        ranked: Sequence[Candidate],
        record_images: Sequence[Image],
        tally: _Tally,
    ) -> list[tuple[Candidate, int]]:
        """T2b: drop files near-identical to an image of the Joconde record; keep the distance."""
        hashes = [phash(j) for i in record_images if (j := self._pop_jpeg(i)) is not None]
        if not hashes:
            logger.warning("%s: no Joconde image could be fetched; skipped", entity_id)
            tally.rejects["joconde_image_missing"] += len(ranked)
            return []
        kept: list[tuple[Candidate, int]] = []
        for c in ranked[: self._targets.hashed_per_entity]:
            jpeg = self._client.get_jpeg(small_thumb(c.file))
            if jpeg is None:
                tally.rejects[Reject.NO_THUMBNAIL] += 1
                continue
            bits = min(distance(phash(jpeg), h) for h in hashes)
            self.near_copy_distances.append(bits)
            if bits <= self._targets.near_copy_bits:
                tally.rejects[Reject.NEAR_COPY] += 1
                continue
            kept.append((c, bits))
        return kept

    def gather(
        self,
        set_id: CommonsSet,
        kind: Kind,
        pool: Mapping[str, WikidataItem],
        *,
        limit: int | None,
        per_entity: int,
        record_images: Mapping[str, Sequence[Image]] | None = None,
    ) -> tuple[list[CommonsPhoto], SetFetch]:
        """Draw entities from `pool` in seeded order; pick photos until `limit` (None: all)."""
        order = sorted(pool)
        self._rng(set_id.value, "order").shuffle(order)
        tally = _Tally()
        photos: list[CommonsPhoto] = []
        for chunk in batched(order, api.TITLES_PER_CALL):
            if limit is not None and len(photos) >= limit:
                break
            self.load_p18(pool[e] for e in chunk)
            for entity in chunk:
                room = per_entity if limit is None else min(per_entity, limit - len(photos))
                if room <= 0:
                    break
                tally.tried += 1
                ranked = self.candidates(entity, pool[entity], kind, tally)
                distances: dict[str, int] = {}
                if record_images is not None:
                    checked = self.not_copies(entity, ranked, record_images[entity], tally)
                    ranked = [c for c, _ in checked]
                    distances = {c.file.title: bits for c, bits in checked}
                chosen = pick(ranked, room)
                if chosen:
                    tally.entities += 1
                photos.extend(
                    to_photo(c, set_id, entity, pool[entity].qid, distances.get(c.file.title))
                    for c in chosen
                )
            logger.info("%s: %d photos from %d entities so far", set_id, len(photos), tally.tried)
        report = SetFetch(
            set_id=set_id,
            kind=kind,
            photos=len(photos),
            entities=tally.entities,
            tried=tally.tried,
            rejects=dict(sorted(tally.rejects.items())),
        )
        return photos, report


def small_thumb(file: CommonsFile) -> str:
    """A 500 px copy (a size Commons keeps ready) for the copy check; else the original."""
    if "/960px-" in file.thumb_url:
        return file.thumb_url.replace("/960px-", "/500px-")
    return file.thumb_url


def to_photo(
    c: Candidate, set_id: CommonsSet, entity_id: str, qid: str, phash_distance: int | None
) -> CommonsPhoto:
    f = c.file
    return CommonsPhoto(
        image_id=commons_image_id(f.title),
        set_id=set_id,
        entity_id=entity_id,
        wikidata_id=qid,
        file_title=f.title,
        page_url=f.page_url,
        url=f.url,
        thumb_url=f.thumb_url,
        width=f.width,
        height=f.height,
        licence=f.licence,
        licence_url=f.licence_url,
        author=f.author,
        credit=f.credit,
        taken=f.taken,
        latitude=f.latitude,
        longitude=f.longitude,
        camera=f.camera,
        source=c.source,
        category=c.category,
        view=c.view,
        phash_distance=phash_distance,
    )


def find_commons_photos(
    records: Sequence[Record],
    client: WikimediaClient,
    pop_jpeg: PopJpeg,
    targets: Targets | None = None,
    seed: int = SEED,
) -> tuple[list[CommonsPhoto], CommonsReport]:
    """T2, T2b and T8c photos for the record table, and what was left out and why."""
    targets = targets or Targets()
    entity_ids = {r.entity_id for r in records}
    buildings = {
        r.entity_id
        for r in records
        if r.collection == Collection.MEMOIRE and r.entity_id.startswith("PA")
    }
    works: dict[str, tuple[Image, ...]] = {
        r.entity_id: r.images for r in records if r.collection == Collection.JOCONDE
    }
    merimee = api.items_with(client, api.MERIMEE)
    joconde = api.items_with(client, api.JOCONDE)
    t2_pool, t2_dropped = unambiguous(merimee, buildings.__contains__)
    t2b_pool, t2b_dropped = unambiguous(joconde, works.__contains__)
    t8_buildings, _ = unambiguous(merimee, lambda i: i.startswith("PA") and i not in entity_ids)
    t8_works, _ = unambiguous(joconde, lambda i: i not in entity_ids)
    logger.info(
        "eligible entities: T2 %d, T2b %d, T8c %d buildings and %d works",
        len(t2_pool),
        len(t2b_pool),
        len(t8_buildings),
        len(t8_works),
    )

    finder = CommonsFinder(client, pop_jpeg, targets, seed)
    t2, t2_report = finder.gather(
        CommonsSet.T2, Kind.BUILDING, t2_pool, limit=targets.t2, per_entity=targets.per_entity
    )
    t2b, t2b_report = finder.gather(
        CommonsSet.T2B,
        Kind.ARTWORK,
        t2b_pool,
        limit=targets.t2b,
        per_entity=targets.per_entity,
        record_images=works,
    )
    t8b, t8b_report = finder.gather(
        CommonsSet.T8C, Kind.BUILDING, t8_buildings, limit=targets.t8_buildings, per_entity=1
    )
    t8w, t8w_report = finder.gather(
        CommonsSet.T8C, Kind.ARTWORK, t8_works, limit=targets.t8_artworks, per_entity=1
    )
    photos, shared = drop_shared([*t2, *t2b, *t8b, *t8w])
    report = CommonsReport(
        sets=(t2_report, t2b_report, t8b_report, t8w_report),
        ambiguous={"T2": t2_dropped, "T2b": t2b_dropped},
        near_copy_distances=tuple(finder.near_copy_distances),
        requests=client.requests,
        shared=shared,
    )
    return photos, report


def drop_shared(photos: Sequence[CommonsPhoto]) -> tuple[list[CommonsPhoto], int]:
    """Drop files chosen for two entities: the photo shows both, so neither is the answer."""
    entities: defaultdict[str, set[str]] = defaultdict(set)
    for p in photos:
        entities[p.image_id].add(p.entity_id)
    shared = {i for i, e in entities.items() if len(e) > 1}
    seen: set[str] = set()
    kept: list[CommonsPhoto] = []
    for p in photos:
        if p.image_id not in shared and p.image_id not in seen:
            seen.add(p.image_id)
            kept.append(p)
    if shared:
        logger.info("dropped %d file(s) chosen for two entities", len(shared))
    return kept, len(shared)


def fetch_commons(settings: Settings, targets: Targets | None = None) -> CommonsReport:
    """Find the Commons photos for the current record table and write `commons.parquet`."""
    records = read_records(settings.records_parquet, settings.images_parquet)
    cache = ImageCache(settings.image_cache_dir, settings.image_cache_mb * 1_000_000)
    with open_client() as wikimedia, httpx.Client(timeout=60.0) as pop:
        fetcher = ImageFetcher(pop, cache)
        client = WikimediaClient(wikimedia, settings.commons_cache_dir)
        photos, report = find_commons_photos(
            records, client, lambda image: fetcher.fetch(image).jpeg, targets
        )
    write_commons(photos, settings.commons_parquet)
    for s in report.sets:
        logger.info(
            "%s %s: %d photos of %d entities (%d looked at); left out: %s",
            s.set_id,
            s.kind,
            s.photos,
            s.entities,
            s.tried,
            ", ".join(f"{k} {v}" for k, v in s.rejects.items()),
        )
    logger.info("%d requests sent; the rest came from the cache", report.requests)
    return report
