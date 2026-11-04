"""The site's ranking on stored vectors, for measuring a built bundle: no model runs.

`SiteBundle` reads a bundle (`miroir.web.bundle`) as the server does. `query_sets` gathers the
test queries it is scored on, with their stored vectors (whole, as the models give them),
their chroma spread and what each may not find. `site_search` ranks them as `Engine.search`
ranks a photo: the same routing (`routing.route`) and fusion (`ranking.rank_photo`), the same
MegaLoc store (whole, PCA-reduced or shortlisted). Everything measured on a bundle (the
build report, the cut-offs, card levels, the sure margin, re-ranking, detection, round R4)
goes through it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

from miroir.config import Settings
from miroir.eval.scoring import first_rank
from miroir.eval.sets import Query, cluster_of, load_queries
from miroir.eval.vectors import VectorSet, read_vectors
from miroir.web import ranking
from miroir.web.catalog import CATALOG_FILE, Catalog, read_catalog
from miroir.web.engine import A3_FILE, A7_FILE, Projection, read_projection
from miroir.web.routing import (
    GUESS_FILE,
    ROUTING_FILE,
    Guess,
    PhotoType,
    Routing,
    read_guess,
    read_routing,
    route,
)
from miroir.web.store import ExactStore, read_exact, shortlist_cosines

if TYPE_CHECKING:
    from miroir.web.bundle import IndexSpec

logger = logging.getLogger(__name__)

IDS_FILE = "ids.txt"
#: Gallery rows cast to float32 at once.
GALLERY_CHUNK = 8192
#: Queries searched at once by `site_search` (two float32 rows of every image each).
SEARCH_BATCH = 256
#: Sets whose queries are gallery images (asked with the gallery's own vectors), and sets of
#: Commons photos (asked with a query run of their own).
POP_SETS = ("T1", "T3", "T5", "T8")
COMMONS_SETS = ("T2", "T2b", "T8c")

Floats = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class BundleVectors:
    ids: tuple[str, ...]
    a7: NDArray[np.float16]
    a3: NDArray[np.float16]
    """MegaLoc as the GPU holds it: whole, or reduced when `projection` is set."""
    projection: Projection | None = None
    exact: ExactStore | None = None
    """Shortlist bundles: every MegaLoc vector whole, in int8 (`miroir.web.store`)."""

    @classmethod
    def load(cls, folder: Path) -> BundleVectors:
        ids = tuple((folder / IDS_FILE).read_text(encoding="utf-8").splitlines())
        return cls(
            ids=ids,
            a7=np.load(folder / A7_FILE, mmap_mode="r"),
            a3=np.load(folder / A3_FILE, mmap_mode="r"),
            projection=read_projection(folder),
            exact=read_exact(folder, mmap=True),
        )

    def searched_a3(self, a3: Floats) -> Floats:
        """Whole unit MegaLoc query rows in the space `a3` is stored in."""
        if self.projection is None or not len(a3):
            return a3
        return np.stack([self.projection.apply(v) for v in a3])


@dataclass(frozen=True, slots=True)
class SiteBundle:
    """A built bundle, read for measuring: the site's ranking without the models."""

    catalog: Catalog
    vectors: BundleVectors
    guess: Guess
    routing: Routing

    @classmethod
    def load(cls, folder: Path) -> SiteBundle:
        return cls(
            catalog=read_catalog(folder / CATALOG_FILE),
            vectors=BundleVectors.load(folder),
            guess=read_guess(folder / GUESS_FILE),
            routing=read_routing(folder / ROUTING_FILE),
        )


@dataclass(frozen=True, slots=True)
class AskedSet:
    """One test set's queries as a bundle is scored on them. The vectors are whole unit rows,
    as the models give them; `site_search` projects MegaLoc for a reduced store."""

    set_id: str
    query_ids: tuple[str, ...]
    query_image_ids: tuple[str, ...]
    a7: Floats
    a3: Floats
    hidden: tuple[tuple[str, ...], ...]
    """Per query, the images it may not find. A not-found query (no right answer) hides
    every image of its entity in the bundle, Commons photos added since the build too."""
    right: tuple[tuple[str, ...], ...]
    chroma: tuple[float, ...]
    """Each query photo's chroma spread (`images.colour`), as the server measures it."""
    clusters: tuple[str, ...] = ()
    """Per query, what it is resampled with (`sets.cluster_of`)."""


def asked_set(
    set_id: str,
    queries: Sequence[Query],
    runs: tuple[VectorSet, VectorSet],
    *,
    catalog: Catalog,
    chroma: Mapping[str, float],
) -> AskedSet:
    """The queries with their vectors from the (A7, A3) `runs`, whole unit rows, and what
    each hides."""
    a7, a3 = runs
    row7, row3 = a7.index(), a3.index()
    v7 = np.asarray(a7.vectors[[row7[q.query_image_id] for q in queries]], dtype=np.float32)
    v3 = np.asarray(a3.vectors[[row3[q.query_image_id] for q in queries]], dtype=np.float32)
    return AskedSet(
        set_id=set_id,
        query_ids=tuple(q.query_id for q in queries),
        query_image_ids=tuple(q.query_image_id for q in queries),
        a7=v7.reshape(len(queries), -1),
        a3=v3.reshape(len(queries), -1),
        hidden=tuple(_hidden(q, catalog) for q in queries),
        right=tuple(q.expected_entity_ids for q in queries),
        chroma=tuple(chroma[q.query_image_id] for q in queries),
        clusters=tuple(cluster_of(q) for q in queries),
    )


def _hidden(query: Query, catalog: Catalog) -> tuple[str, ...]:
    hidden = (query.query_image_id, *query.exclude_image_ids)
    entity = catalog.entity_row.get(query.entity_id)
    if query.expected_entity_ids or entity is None:
        return hidden
    rows = catalog.entity_image_rows(entity)
    return tuple(dict.fromkeys((*hidden, *(catalog.images[int(r)].image_id for r in rows))))


def query_sets(
    catalog: Catalog,
    queries: Mapping[str, Sequence[Query]],
    *,
    gallery: tuple[VectorSet, VectorSet],
    commons: tuple[VectorSet, VectorSet] | None,
    chroma: Mapping[str, float],
) -> dict[str, AskedSet]:
    """Each set's queries the bundle can ask: POP sets with the gallery runs' rows of their
    query images (in the bundle), Commons sets with their query runs (when given)."""
    out: dict[str, AskedSet] = {}
    in_bundle = set(catalog.image_row)
    for set_id, asked in queries.items():
        runs = commons if set_id in COMMONS_SETS else gallery
        if runs is None:
            logger.info("%s: no query vectors for Commons photos; not scored", set_id)
            continue
        have = set(runs[0].ids) & set(runs[1].ids)
        if runs is gallery:
            have &= in_bundle
        kept = [q for q in asked if q.query_image_id in have]
        if len(kept) < len(asked):
            logger.info(
                "%s: %d of %d queries have no vector", set_id, len(asked) - len(kept), len(asked)
            )
        out[set_id] = asked_set(set_id, kept, runs, catalog=catalog, chroma=chroma)
    return out


def tune_sets(settings: Settings, spec: IndexSpec, catalog: Catalog) -> dict[str, AskedSet]:
    """The tune queries a bundle's report scores (T1, T3, T5, T8; T2, T2b, T8c with query runs)."""
    from miroir.eval.router3 import query_chroma  # noqa: PLC0415 (fetches photos)

    commons = (
        None
        if not (spec.a7_query_run and spec.a3_query_run)
        else (
            read_vectors(settings.vectors_path(spec.a7_query_run)),
            read_vectors(settings.vectors_path(spec.a3_query_run)),
        )
    )
    set_ids = (*POP_SETS, *(COMMONS_SETS if commons else ()))
    queries = {s: load_queries(settings, s, "tune") for s in set_ids}
    photos = list(dict.fromkeys(q.query_image_id for qs in queries.values() for q in qs))
    gallery = (
        read_vectors(settings.vectors_path(spec.a7_run)),
        read_vectors(settings.vectors_path(spec.a3_run)),
    )
    return query_sets(
        catalog, queries, gallery=gallery, commons=commons, chroma=query_chroma(settings, photos)
    )


@dataclass(frozen=True, slots=True)
class SiteHit:
    """One photo searched as the site searches it (no filters)."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    bucket: str
    photo_type: PhotoType
    entities: tuple[str, ...]
    """The top entities, best first."""
    confidences: tuple[float, ...]
    """Per top entity: the routed mix of its raw cosines (what "not found" reads)."""
    best_images: tuple[str, ...]
    """Per top entity: its image the query matches best (what a card shows)."""
    right_rank: int
    """Rank of the first right entity over every entity, a tie with a wrong one counted
    against it (`scoring.first_rank`); 0 when no right entity has an image left to find."""


def site_search(bundle: SiteBundle, asked: AskedSet, *, top: int) -> list[SiteHit]:
    """`Engine.search` on the queries' stored vectors, with each query's images hidden."""
    catalog, vectors = bundle.catalog, bundle.vectors
    n = vectors.a7.shape[0]
    out: list[SiteHit] = []
    for start in range(0, len(asked.query_ids), SEARCH_BATCH):
        stop = min(start + SEARCH_BATCH, len(asked.query_ids))
        whole = asked.a3[start:stop]
        s7 = _image_cosines(asked.a7[start:stop], vectors.a7)
        s3 = _image_cosines(vectors.searched_a3(whole), vectors.a3)
        for j in range(stop - start):
            q = start + j
            shown = np.ones(n, dtype=np.bool_)
            shown[[catalog.image_row[i] for i in asked.hidden[q] if i in catalog.image_row]] = False
            a7 = np.where(shown, s7[j], -np.inf).astype(np.float32)
            a3 = np.where(shown, s3[j], -np.inf).astype(np.float32)
            if vectors.exact is not None:
                a3 = shortlist_cosines(vectors.exact, whole[j], a7, a3)
            photo_type, bucket = route(bundle.routing, bundle.guess, asked.a7[q], asked.chroma[q])
            ranked = ranking.rank_photo(
                ranking.entity_max(a7, catalog), ranking.entity_max(a3, catalog), bucket.a7_weight
            )
            best = ranking.top_entities(ranked.fused, top)
            per_image = ranking.weighted(a7, a3, bucket.a7_weight)
            images: list[str] = []
            for e in best:
                found = ranking.best_image(catalog.entity_image_rows(int(e)), per_image, shown)
                images.append("" if found is None else catalog.images[found].image_id)
            # Right entities with an image left to find (EmbeddingGemma 2 scores every image).
            # One the ranking leaves out (-inf: off a MegaLoc shortlist) is a miss.
            wanted = [
                r
                for e in asked.right[q]
                if (r := catalog.entity_row.get(e)) is not None and np.isfinite(ranked.e7[r])
            ]
            out.append(
                SiteHit(
                    bucket=bucket.name,
                    photo_type=photo_type,
                    entities=tuple(catalog.entity_ids[int(e)] for e in best),
                    confidences=tuple(ranked.confidence(int(e)) for e in best),
                    best_images=tuple(images),
                    right_rank=first_rank(ranked.fused, wanted),
                )
            )
    return out


def entity_cosines(
    queries: Floats,
    gallery: NDArray[np.float16],
    catalog: Catalog,
    hidden: Sequence[Sequence[str]],
) -> Floats:
    """Best cosine per entity for each query, its hidden images removed. Chunked so a
    137,000 x 8,448 gallery never sits in memory as float32 all at once."""
    out = np.empty((len(queries), catalog.entity_count), dtype=np.float32)
    for start in range(0, len(queries), SEARCH_BATCH):
        sims = _image_cosines(queries[start : start + SEARCH_BATCH], gallery)
        for j in range(len(sims)):
            rows = [catalog.image_row[i] for i in hidden[start + j] if i in catalog.image_row]
            sims[j, rows] = -np.inf
            out[start + j] = ranking.entity_max(sims[j], catalog)
    return out


def _image_cosines(queries: Floats, gallery: NDArray[np.float16]) -> Floats:
    sims = np.empty((len(queries), gallery.shape[0]), dtype=np.float32)
    for g in range(0, gallery.shape[0], GALLERY_CHUNK):
        block = np.asarray(gallery[g : g + GALLERY_CHUNK], dtype=np.float32)
        sims[:, g : g + len(block)] = queries @ block.T
    return sims
