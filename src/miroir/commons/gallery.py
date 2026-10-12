"""F2: Commons photos of Mérimée buildings as extra gallery images (experiments.md step F).

Mémoire holds mostly black-and-white archive prints; a visitor sends a colour phone photo.
F2 adds, for each building of a gallery, up to `PER_BUILDING` recent Commons photos of it,
found as T2's are (Wikidata Mérimée id P380 → P18 and Commons category P373, the same
`photo_problem` rules), exteriors first, one per visit first. Each counts as an image of the
building's entity.

No test query may reach the gallery this way. A candidate is left out when it is:

- a Commons query file (T2, T2b, T8c, tune or sealed: every photo of `commons.parquet`);
- within `NEAR_COPY_BITS` of a query photo (perceptual hash on the 512 px copies), as T2b
  leaves out the museum's own reproduction;
- by the same author on the same day as a query photo of the same building: another frame
  of that visit;
- within `NEAR_COPY_BITS` of a POP image of the same building, so no T1 query finds a copy
  of itself.

Photos are read as the 500 px copies Commons keeps ready (`build.small_thumb`): faster to
serve than the 960 px ones queries use, and the models see 512 px at most. Every answer is
cached (`WikimediaClient`), so a rerun sends nothing. The same code runs for
the full sample (`gallery="full"`); `FetchReport.estimate` says what that costs in requests.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import httpx

from miroir.commons import api
from miroir.commons.api import CommonsFile, WikidataItem
from miroir.commons.build import NEAR_COPY_BITS, drop_shared, small_thumb, to_photo, unambiguous
from miroir.commons.client import MAX_PER_SECOND, WikimediaClient, open_client
from miroir.commons.images import commons_photos
from miroir.commons.model import CommonsPhoto, CommonsSet, Source, View
from miroir.commons.select import (
    Candidate,
    Kind,
    names_place,
    only_commune,
    photo_problem,
    rank,
    view_of,
)
from miroir.config import SEED, Settings
from miroir.eval.sets import read_gallery, write_gallery
from miroir.images.fetch import ImageCache, ImageFetcher
from miroir.images.phash import distance, phash
from miroir.records.model import Image
from miroir.records.store import read_images, read_records, write_commons

logger = logging.getLogger(__name__)

#: Commons photos added per building at most.
PER_BUILDING = 5


class Excluded(StrEnum):
    """Why a candidate that passed `photo_problem` is still not added."""

    UNNAMED = "unnamed"
    """A category file whose title shares no distinctive word with the category name
    (label check, docs/results.md, filter A). P18 files are kept."""
    ONLY_COMMUNE = "only_commune"
    """A title holding only the commune and numbers (filter B)."""
    QUERY_FILE = "query_file"
    QUERY_COPY = "query_copy"
    QUERY_VISIT = "query_visit"
    POP_COPY = "pop_copy"
    NO_THUMBNAIL = "no_thumbnail"


@dataclass(frozen=True, slots=True)
class FetchReport:
    gallery: str
    buildings: int
    """Mérimée buildings (PA entities) with an image in the gallery."""
    with_item: int
    """Of those, buildings with one unambiguous Wikidata item holding a photo or a category."""
    with_photos: int
    photos: int
    rejects: Mapping[str, int]
    """Candidates left out, by reason (`select.Reject` and `Excluded`)."""
    requests: int
    """Requests sent to Wikimedia in this run; 0 when every answer came from the cache."""
    thumbnails: int
    """Thumbnails read (from the cache or Wikimedia) to check copies."""

    def estimate(self, buildings: int) -> tuple[int, float]:
        """Requests and hours, with nothing cached, to run this for `buildings` buildings."""
        share = self.with_item / max(self.buildings, 1)
        per_item = (self.thumbnails / max(self.with_item, 1)) + 1  # thumbnails + one listing
        requests = round(buildings * share * per_item + buildings / api.TITLES_PER_CALL + 1)
        return requests, requests / MAX_PER_SECOND / 3600


@dataclass(frozen=True, slots=True)
class Guard:
    """What a candidate is checked against: every query photo, and the building's POP images."""

    titles: frozenset[str]
    hashes: tuple[int, ...]
    visits: Mapping[str, frozenset[tuple[str, str]]]
    """Per building, (author, day) of its query photos."""


def query_guard(
    queries: Iterable[CommonsPhoto], jpeg_of: Callable[[CommonsPhoto], bytes | None]
) -> Guard:
    """Hash every query photo (all sets, both halves) once. Fails if one cannot be read."""
    photos = list(queries)
    hashes: list[int] = []
    missing: list[str] = []
    visits: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    for photo in photos:
        visits[photo.entity_id].add((photo.author, photo.taken[:10]))
        jpeg = jpeg_of(photo)
        if jpeg is None:
            missing.append(photo.file_title)
        else:
            hashes.append(phash(jpeg))
    if missing:
        msg = f"{len(missing)} query photo(s) could not be read for the copy check: {missing[:3]}"
        raise RuntimeError(msg)
    logger.info("copy check: %d query photos hashed", len(hashes))
    return Guard(
        titles=frozenset(p.file_title for p in photos),
        hashes=tuple(hashes),
        visits={e: frozenset(v) for e, v in visits.items()},
    )


def order_for_gallery(ranked: Sequence[Candidate]) -> list[Candidate]:
    """Exteriors first, then one file per visit before a second of the same visit.

    `ranked` comes from `select.rank` (confirmed and named files first, seeded order).
    """
    exteriors = [c for c in ranked if c.view == View.EXTERIOR]
    others = [c for c in ranked if c.view != View.EXTERIOR]
    seen: set[tuple[str, str]] = set()
    first: list[Candidate] = []
    later: list[Candidate] = []
    for c in (*exteriors, *others):
        (later if c.visit in seen else first).append(c)
        seen.add(c.visit)
    return first + later


class GalleryFinder:
    """Picks up to `per_building` Commons photos per building that pass every check."""

    def __init__(
        self,
        client: WikimediaClient,
        guard: Guard,
        pop_hashes: Callable[[str], Sequence[int]],
        *,
        per_building: int = PER_BUILDING,
        near_copy_bits: int = NEAR_COPY_BITS,
        seed: int = SEED,
    ) -> None:
        self._client = client
        self.guard = guard
        self._pop_hashes = pop_hashes
        self._per_building = per_building
        self._bits = near_copy_bits
        self._seed = seed
        self.rejects: Counter[str] = Counter()
        """Candidates left out, by reason."""
        self.thumbnails = 0
        """Thumbnails read to check copies."""

    def candidates(
        self,
        entity_id: str,
        item: WikidataItem,
        p18: Mapping[str, CommonsFile],
        commune: str = "",
    ) -> list[Candidate]:
        """P18 and the newest files of the item's categories that look like a recent photo.

        A category file must also name the place (filter A) and hold more than the commune
        (filter B): the label check found every wrong T2 label among category files.
        """
        found: list[tuple[CommonsFile, Source, str]] = [
            (p18[t], Source.P18, "") for t in item.images if t in p18
        ]
        for category in item.categories:
            listing = api.category_files(self._client, category)
            found.extend((f, Source.CATEGORY, category) for f in listing.files)
        kept: list[Candidate] = []
        for file, source, category in found:
            problem = photo_problem(file, entity_id, Kind.BUILDING)
            if problem is not None:
                self.rejects[problem] += 1
                continue
            if file.title in self.guard.titles:
                self.rejects[Excluded.QUERY_FILE] += 1
                continue
            named = any(names_place(file, c) for c in item.categories)
            if source is Source.CATEGORY and not named:
                self.rejects[Excluded.UNNAMED] += 1
                continue
            if source is Source.CATEGORY and only_commune(file, commune):
                self.rejects[Excluded.ONLY_COMMUNE] += 1
                continue
            kept.append(Candidate(file, source, category, view_of(file), named))
        rng = random.Random(":".join((str(self._seed), "commons-gallery", entity_id)))
        return order_for_gallery(rank(kept, rng))

    def pick(self, entity_id: str, ordered: Sequence[Candidate]) -> list[tuple[Candidate, int]]:
        """The first `per_building` candidates that are no copy and no frame of a query visit.

        Returns each with its distance in bits to the nearest query photo.
        """
        visits = self.guard.visits.get(entity_id, frozenset())
        own = self._pop_hashes(entity_id)
        chosen: list[tuple[Candidate, int]] = []
        for c in ordered:
            if len(chosen) >= self._per_building:
                break
            if c.visit in visits:
                self.rejects[Excluded.QUERY_VISIT] += 1
                continue
            jpeg = self._client.get_jpeg(small_thumb(c.file))
            self.thumbnails += 1
            if jpeg is None:
                self.rejects[Excluded.NO_THUMBNAIL] += 1
                continue
            h = phash(jpeg)
            nearest = min((distance(h, q) for q in self.guard.hashes), default=64)
            if nearest <= self._bits:
                self.rejects[Excluded.QUERY_COPY] += 1
                continue
            if any(distance(h, p) <= self._bits for p in own):
                self.rejects[Excluded.POP_COPY] += 1
                continue
            chosen.append((c, nearest))
        return chosen


def find_gallery_photos(
    buildings: Sequence[str],
    client: WikimediaClient,
    finder: GalleryFinder,
    *,
    gallery: str,
    communes: Mapping[str, str],
) -> tuple[list[CommonsPhoto], FetchReport]:
    """Commons photos for each building, and counts of what was found and left out."""
    wanted = set(buildings)
    pool, ambiguous = unambiguous(api.items_with(client, api.MERIMEE), wanted.__contains__)
    logger.info(
        "%d buildings: %d with a Wikidata item, %d ambiguous",
        len(wanted),
        len(pool),
        ambiguous,
    )
    p18 = api.files_by_title(client, api.titles(list(pool.values())))
    photos: list[CommonsPhoto] = []
    with_photos = 0
    for n, entity in enumerate(sorted(pool), start=1):
        item = pool[entity]
        chosen = finder.pick(entity, finder.candidates(entity, item, p18, communes.get(entity, "")))
        with_photos += bool(chosen)
        photos.extend(
            replace(
                to_photo(c, CommonsSet.GALLERY, entity, item.qid, d), thumb_url=small_thumb(c.file)
            )
            for c, d in chosen
        )
        if n % 100 == 0:
            logger.info(
                "%d/%d buildings, %d photos, %d requests",
                n,
                len(pool),
                len(photos),
                client.requests,
            )
    photos, shared = drop_shared(photos)
    rejects = Counter(finder.rejects)
    if shared:
        rejects["shared"] = shared
    report = FetchReport(
        gallery=gallery,
        buildings=len(wanted),
        with_item=len(pool),
        with_photos=with_photos,
        photos=len(photos),
        rejects=dict(sorted(rejects.items())),
        requests=client.requests,
        thumbnails=finder.thumbnails,
    )
    return photos, report


def gallery_buildings(
    images: Iterable[Image], gallery_ids: Iterable[str]
) -> dict[str, list[Image]]:
    """Mérimée buildings (PA entities) with an image in the gallery, with those images."""
    of_id = {i.image_id: i for i in images}
    out: defaultdict[str, list[Image]] = defaultdict(list)
    for image_id in gallery_ids:
        image = of_id.get(image_id)
        if image is not None and image.entity_id.startswith("PA"):
            out[image.entity_id].append(image)
    return dict(out)


def building_communes(settings: Settings) -> dict[str, str]:
    """Each Mérimée building's commune, the one its records name most often."""
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in read_records(settings.records_parquet, settings.images_parquet):
        if record.entity_id.startswith("PA") and record.commune:
            counts[record.entity_id][record.commune] += 1
    return {e: c.most_common(1)[0][0] for e, c in counts.items()}


def fetch_gallery_photos(settings: Settings, gallery: str) -> FetchReport:
    """Find F2 photos for the buildings of `gallery`; write `commons-gallery-<gallery>.parquet`.

    Also writes the gallery list `<gallery>+F2`: the gallery plus every photo found.
    """
    gallery_ids = read_gallery(settings, gallery)
    buildings = gallery_buildings(read_images(settings.images_parquet), gallery_ids)
    queries = list(commons_photos(settings).values())
    if not queries:
        msg = "no commons.parquet: run `miroir testsets commons` first (the copy check needs it)"
        raise FileNotFoundError(msg)
    cache = ImageCache(settings.image_cache_dir, settings.image_cache_mb * 1_000_000)
    with open_client() as wikimedia, httpx.Client(timeout=60.0) as pop:
        client = WikimediaClient(wikimedia, settings.commons_cache_dir)
        fetcher = ImageFetcher(pop, cache)
        guard = query_guard(queries, lambda p: client.get_jpeg(p.thumb_url))
        hashed: dict[str, list[int]] = {}

        def pop_hashes(entity: str) -> list[int]:
            if entity not in hashed:
                found = (fetcher.fetch(i).jpeg for i in buildings.get(entity, []))
                hashed[entity] = [phash(j) for j in found if j is not None]
            return hashed[entity]

        finder = GalleryFinder(client, guard, pop_hashes)
        photos, report = find_gallery_photos(
            sorted(buildings), client, finder, gallery=gallery, communes=building_communes(settings)
        )
    write_commons(photos, settings.commons_gallery_parquet(gallery))
    write_gallery(settings, f"{gallery}+F2", [*gallery_ids, *(p.image_id for p in photos)])
    logger.info(
        "%s: %d photos for %d of %d buildings (%d with a Wikidata item); left out: %s; "
        "%d requests sent",
        gallery,
        report.photos,
        report.with_photos,
        report.buildings,
        report.with_item,
        ", ".join(f"{k} {v}" for k, v in report.rejects.items()),
        report.requests,
    )
    return report
