"""The site's ranking on stored vectors, for measuring a built bundle: no model runs.

`SiteBundle` reads a bundle (`miroir.web.bundle`) as the server does. `QuerySets` loads the
test queries a bundle is scored on, with their stored vectors and what each may not find.
`search_set` and `site_search` rank them as `Engine.search` ranks a photo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

from miroir.config import Settings
from miroir.eval.scoring import first_rank
from miroir.eval.sets import Query, load_queries
from miroir.eval.vectors import VectorSet, read_vectors
from miroir.web import ranking
from miroir.web.catalog import CATALOG_FILE, Catalog, read_catalog
from miroir.web.engine import A3_FILE, A7_FILE, Projection, read_projection, weighted
from miroir.web.routing import (
    GUESS_FILE,
    ROUTING_FILE,
    Guess,
    PhotoType,
    Routing,
    read_guess,
    read_routing,
)
from miroir.web.store import ExactStore, read_exact, shortlist_cosines

if TYPE_CHECKING:
    from miroir.web.bundle import IndexSpec

IDS_FILE = "ids.txt"
#: Queries searched at once when scoring, and gallery rows cast to float32 at once.
QUERY_BATCH = 128
GALLERY_CHUNK = 8192
#: Queries searched at once by `site_search` (two float32 rows of every image each).
SEARCH_BATCH = 256

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

    def a3_entity_cosines(
        self, catalog: Catalog, a7: Floats, a3: Floats, hidden: Sequence[Sequence[str]]
    ) -> Floats:
        """Best MegaLoc cosine per entity, as the site computes it from this bundle: whole,
        reduced, or exact over the shortlist. `a3` holds whole (8,448-d) unit query vectors."""
        reduced = a3
        if self.projection is not None and len(a3):
            reduced = np.stack([self.projection.apply(v) for v in a3])
        if self.exact is None:
            return entity_cosines(reduced, self.a3, catalog, hidden)
        out = np.empty((len(a3), catalog.entity_count), dtype=np.float32)
        for start in range(0, len(a3), QUERY_BATCH):
            stop = min(start + QUERY_BATCH, len(a3))
            s7 = _image_cosines(a7[start:stop], self.a7)
            s3 = _image_cosines(reduced[start:stop], self.a3)
            for j in range(stop - start):
                rows = [catalog.image_row[i] for i in hidden[start + j] if i in catalog.image_row]
                s7[j, rows] = -np.inf
                s3[j, rows] = -np.inf
                exact = shortlist_cosines(self.exact, a3[start + j], s7[j], s3[j])
                out[start + j] = ranking.entity_max(exact, catalog)
        return out


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
    """One test set's tune queries as `build-index` scores them: vectors, hidden, answers."""

    set_id: str
    query_ids: tuple[str, ...]
    query_image_ids: tuple[str, ...]
    a7: Floats
    a3: Floats
    hidden: tuple[tuple[str, ...], ...]
    right: tuple[tuple[str, ...], ...]
    chroma: tuple[float, ...]
    """Each query photo's chroma spread (`images.colour`), as the server measures it."""


@dataclass(frozen=True, slots=True)
class QuerySets:
    sets: dict[str, AskedSet]

    @classmethod
    def load(
        cls, settings: Settings, spec: IndexSpec, ids: Sequence[str], projection: Projection | None
    ) -> QuerySets:
        from miroir.eval.router3 import query_chroma  # noqa: PLC0415 (fetches photos)

        index = set(ids)
        gallery_a7 = read_vectors(settings.vectors_path(spec.a7_run))
        gallery_a3 = read_vectors(settings.vectors_path(spec.a3_run))
        chosen: dict[str, tuple[list[Query], VectorSet, VectorSet, Projection | None]] = {}
        for set_id in ("T1", "T3", "T5", "T8"):
            queries = [
                q for q in load_queries(settings, set_id, "tune") if q.query_image_id in index
            ]
            chosen[set_id] = (queries, gallery_a7, gallery_a3, None)
        if spec.a7_query_run and spec.a3_query_run:
            q7 = read_vectors(settings.vectors_path(spec.a7_query_run))
            q3 = read_vectors(settings.vectors_path(spec.a3_query_run))
            have = set(q7.ids) & set(q3.ids)
            for set_id in ("T2", "T2b", "T8c"):
                queries = [
                    q for q in load_queries(settings, set_id, "tune") if q.query_image_id in have
                ]
                chosen[set_id] = (queries, q7, q3, projection)
        photos = [q.query_image_id for qs, *_ in chosen.values() for q in qs]
        chroma = query_chroma(settings, list(dict.fromkeys(photos)))
        return cls(
            {s: _asked(s, qs, v7, v3, p, chroma=chroma) for s, (qs, v7, v3, p) in chosen.items()}
        )


def _asked(
    set_id: str,
    queries: Sequence[Query],
    a7: VectorSet,
    a3: VectorSet,
    projection: Projection | None,
    *,
    chroma: Mapping[str, float],
) -> AskedSet:
    row7, row3 = a7.index(), a3.index()
    v7 = np.asarray(a7.vectors[[row7[q.query_image_id] for q in queries]], dtype=np.float32)
    v3 = np.asarray(a3.vectors[[row3[q.query_image_id] for q in queries]], dtype=np.float32)
    if projection is not None:
        v3 = np.stack([projection.apply(v) for v in v3]) if len(v3) else v3
    return AskedSet(
        set_id=set_id,
        query_ids=tuple(q.query_id for q in queries),
        query_image_ids=tuple(q.query_image_id for q in queries),
        a7=v7,
        a3=v3,
        hidden=tuple((q.query_image_id, *q.exclude_image_ids) for q in queries),
        right=tuple(q.expected_entity_ids for q in queries),
        chroma=tuple(chroma[q.query_image_id] for q in queries),
    )


def asked_sets(settings: Settings, spec: IndexSpec) -> dict[str, AskedSet]:
    """The tune queries the bundle of `spec` is scored on (T1, T3, T5, T8, T2, T2b, T8c)."""
    folder = settings.web_index_dir(spec.name)
    ids = (folder / IDS_FILE).read_text(encoding="utf-8").splitlines()
    return QuerySets.load(settings, spec, ids, read_projection(folder)).sets


@dataclass(frozen=True, slots=True)
class Searched:
    """Per query: rank of the first right entity (0 when none is in the index), the top
    entity's confidence, and the bucket the query went to."""

    ranks: NDArray[np.int64]
    confidence: NDArray[np.float64]
    buckets: tuple[str, ...]


def search_set(
    catalog: Catalog, vectors: BundleVectors, guess: Guess, routing: Routing, asked: AskedSet
) -> Searched:
    e7 = entity_cosines(asked.a7, vectors.a7, catalog, asked.hidden)
    e3 = vectors.a3_entity_cosines(catalog, asked.a7, asked.a3, asked.hidden)
    ranks: list[int] = []
    confidence: list[float] = []
    buckets: list[str] = []
    for j in range(len(asked.query_ids)):
        photo_type, _ = guess.photo_type(asked.a7[j])
        bucket = routing.choose(photo_type, asked.chroma[j])
        w = bucket.a7_weight
        signals = [
            s for s in (ranking.Signal(e7[j], w), ranking.Signal(e3[j], 1 - w)) if s.weight > 0
        ]
        fused = ranking.fuse(signals)
        top = int(np.argmax(np.where(np.isfinite(fused), fused, -np.inf)))
        confidence.append(float(weighted(e7[j, top : top + 1], e3[j, top : top + 1], w)[0]))
        buckets.append(bucket.name)
        # Right entities with an image left to find (EmbeddingGemma 2 scores every image).
        right = [
            r
            for e in asked.right[j]
            if (r := catalog.entity_row.get(e)) is not None and np.isfinite(e7[j, r])
        ]
        # A right answer the ranking leaves out (-inf: off a MegaLoc shortlist) is a miss.
        ranks.append(first_rank(fused, right))
    return Searched(np.asarray(ranks), np.asarray(confidence), tuple(buckets))


def entity_cosines(
    queries: Floats,
    gallery: NDArray[np.float16],
    catalog: Catalog,
    hidden: Sequence[Sequence[str]],
) -> Floats:
    """Best cosine per entity for each query, its hidden images removed. Chunked so a
    137,000 x 8,448 gallery never sits in memory as float32 all at once."""
    out = np.empty((len(queries), catalog.entity_count), dtype=np.float32)
    n = gallery.shape[0]
    for start in range(0, len(queries), QUERY_BATCH):
        batch = queries[start : start + QUERY_BATCH]
        sims = np.empty((len(batch), n), dtype=np.float32)
        for g in range(0, n, GALLERY_CHUNK):
            block = np.asarray(gallery[g : g + GALLERY_CHUNK], dtype=np.float32)
            sims[:, g : g + len(block)] = batch @ block.T
        for j in range(len(batch)):
            rows = [catalog.image_row[i] for i in hidden[start + j] if i in catalog.image_row]
            sims[j, rows] = -np.inf
            out[start + j] = ranking.entity_max(sims[j], catalog)
    return out


@dataclass(frozen=True, slots=True)
class SiteHit:
    """One photo searched as the site searches it (no filters, chroma not asked)."""

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
    """Rank of the first right entity over every entity; 0 when none is in the index."""


def site_search(
    bundle: SiteBundle,
    a7: Floats,
    a3: Floats,
    *,
    hidden: Sequence[Sequence[str]],
    right: Sequence[Sequence[str]],
    top: int,
) -> list[SiteHit]:
    """`Engine.search` on given query vectors (unit rows), with images hidden per query."""
    catalog, vectors = bundle.catalog, bundle.vectors
    if vectors.exact is not None:
        msg = "site_search does not shortlist: measure a shortlist bundle with build-index"
        raise ValueError(msg)
    n = vectors.a7.shape[0]
    out: list[SiteHit] = []
    for start in range(0, len(a7), SEARCH_BATCH):
        stop = min(start + SEARCH_BATCH, len(a7))
        s7 = _image_cosines(a7[start:stop], vectors.a7)
        s3 = _image_cosines(a3[start:stop], vectors.a3)
        for j in range(stop - start):
            q = start + j
            rows = [catalog.image_row[i] for i in hidden[q] if i in catalog.image_row]
            s7[j, rows] = -np.inf
            s3[j, rows] = -np.inf
            photo_type, _ = bundle.guess.photo_type(a7[q])
            bucket = bundle.routing.choose(photo_type, None)
            w = bucket.a7_weight
            e7 = ranking.entity_max(s7[j], catalog)
            e3 = ranking.entity_max(s3[j], catalog)
            signals = [
                s for s in (ranking.Signal(e7, w), ranking.Signal(e3, 1 - w)) if s.weight > 0
            ]
            fused = ranking.fuse(signals)
            order = ranking.top_entities(fused, catalog.entity_count)
            best = order[:top]
            shown = np.ones(n, dtype=np.bool_)
            shown[rows] = False
            per_image = np.where(shown, w * s7[j] + (1 - w) * s3[j], -np.inf).astype(np.float32)
            images: list[str] = []
            for e in best:
                found = ranking.best_image(catalog.entity_image_rows(int(e)), per_image, shown)
                images.append("" if found is None else catalog.images[found].image_id)
            wanted = [
                r
                for e in right[q]
                if (r := catalog.entity_row.get(e)) is not None and np.isfinite(e7[r])
            ]
            out.append(
                SiteHit(
                    bucket=bucket.name,
                    photo_type=photo_type,
                    entities=tuple(catalog.entity_ids[int(e)] for e in best),
                    confidences=tuple(float(w * e7[e] + (1 - w) * e3[e]) for e in best),
                    best_images=tuple(images),
                    right_rank=first_rank(fused, wanted),
                )
            )
    return out


def _image_cosines(queries: Floats, gallery: NDArray[np.float16]) -> Floats:
    sims = np.empty((len(queries), gallery.shape[0]), dtype=np.float32)
    for g in range(0, gallery.shape[0], GALLERY_CHUNK):
        block = np.asarray(gallery[g : g + GALLERY_CHUNK], dtype=np.float32)
        sims[:, g : g + len(block)] = queries @ block.T
    return sims
