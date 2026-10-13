"""Build a web index bundle on the Mac: vectors, catalog, type guess, weights and cut-offs.

`miroir web build-index` writes `Settings.web_index_dir(name)`:

| File | Holds |
|---|---|
| `ids.txt`, `a7.npy`, `a3.npy` | image ids, both models' unit vectors (float16), same rows |
| `catalog.json.gz` | the images (row order) and their records, for cards, filters and pages |
| `guess.npz` | G4's building-or-object guess (logistic regression on A7 vectors) |
| `routing.json` | per bucket: A7 weight and "not found" cut-off, with where each came from |
| `report.json` | how the site's ranking scores on the tune halves with this index |
| `manifest.json` | the vector runs used, counts, date (written last: marks a finished bundle) |

The cut-offs come from T8 (technical spec §6.3, §7): T1 and T5 tune queries with their whole
building or object removed from the index, so any answer is wrong. Per bucket, the cut-off
is the lowest confidence that keeps at most `WRONG_TARGET` of those queries confident.
Everything here is NumPy on stored vectors: no model runs.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

from miroir.config import Settings
from miroir.eval import router
from miroir.eval.sets import Query, load_queries
from miroir.eval.vectors import VectorSet, read_vectors
from miroir.records.model import Record
from miroir.records.store import read_commons, read_places, read_records
from miroir.web import ranking
from miroir.web.catalog import (
    CATALOG_FILE,
    Catalog,
    CatalogData,
    CatalogImage,
    CatalogRecord,
    Source,
    read_catalog,
    write_catalog,
)
from miroir.web.engine import (
    A3_FILE,
    A3_PROJECTION_FILE,
    A7_FILE,
    Projection,
    read_projection,
    weighted,
)
from miroir.web.nearby import PLACES_FILE, write_places
from miroir.web.routing import (
    GUESS_FILE,
    ROUTING_FILE,
    Bucket,
    Guess,
    PhotoType,
    Routing,
    read_guess,
    read_routing,
    write_guess,
    write_routing,
)
from miroir.web.store import ExactStore, read_exact, shortlist_cosines, write_exact
from miroir.web.sure import SURE_FILE

logger = logging.getLogger(__name__)

IDS_FILE = "ids.txt"
MANIFEST_FILE = "manifest.json"
REPORT_FILE = "report.json"
#: Most T8 searches allowed to get a confident answer at the cut-off (the brief: about 10%).
WRONG_TARGET = 0.10
#: Characters of a record's description kept for its page.
DESCRIPTION_CHARS = 700
#: Queries searched at once when scoring, and gallery rows cast to float32 at once.
QUERY_BATCH = 128
GALLERY_CHUNK = 8192

#: G5 weights from R2 (docs/results.md, "F2"): with Commons photos in the gallery the grid
#: picked EmbeddingGemma 2 at 1.0 for black-and-white building photos, 0.0 (MegaLoc alone)
#: for colour building photos and 0.2 for objects (hard switch), on T1, T2, T3 and T5 tune.
ARCHIVE_A7_WEIGHT = 1.0
COLOUR_BUILDING_A7_WEIGHT = 0.0
OBJECT_A7_WEIGHT = 0.2
#: A bucket with fewer T8 searches than this takes its cut-off from T8c (Commons photos with
#: no record) instead: T8 is drawn from archive prints and museum photos, so it holds almost
#: no colour photo of a building.
MIN_CALIBRATION_QUERIES = 20

Floats = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """Which vector runs make an index. The runs must hold the same image ids."""

    name: str
    a7_run: str
    a3_run: str
    a7_query_run: str | None = None
    """Commons query photos (T2, T2b, T8c) embedded apart, scored in the report when given."""
    a3_query_run: str | None = None
    saved_guess: str | None = None
    """A building-or-object guess already trained on this index (`miroir index guess`), a file
    in `Settings.index_dir`; else `build-index` trains one (slow on the full sample)."""
    a3_projection: str | None = None
    """A file under the A3 run folder holding `mean` and `components`, when MegaLoc vectors
    are stored reduced (the query vectors are then reduced the same way)."""
    a3_shortlist_dims: int | None = None
    """Store MegaLoc for all of POP (`miroir.web.store`): reduced by PCA to this many dims on
    the GPU, to shortlist images, and whole in int8 in host memory, to score the shortlist.
    None keeps every 8,448-d vector in float16 on the GPU."""


INDEXES: dict[str, IndexSpec] = {
    "screening-v2": IndexSpec(
        name="screening-v2",
        a7_run="A7-embeddinggemma-2-512-screening-v2-publishable",
        a3_run="A3-megaloc-322-screening-v2-publishable",
        a7_query_run="A7-embeddinggemma-2-512-T2-T2b-T8c-tune-queries",
        a3_query_run="A3-megaloc-322-T2-T2b-T8c-tune-queries",
    ),
    # Every image of the July exports the Modal runs could fetch (136,731 of 137,775), from
    # the full-sample index (`miroir embed modal --all-images`); MegaLoc kept at 8,448 dims.
    "full": IndexSpec(
        name="full",
        a7_run="A7-embeddinggemma-2-512-full",
        a3_run="A3-megaloc-322-full",
        a7_query_run="A7-embeddinggemma-2-512-T2-T2b-T8c-tune-queries",
        a3_query_run="A3-megaloc-322-T2-T2b-T8c-tune-queries",
        saved_guess="guess-A7-embeddinggemma-2-512-full.npz",
    ),
    # The same, plus up to 5 Commons photos per Mérimée building (step F2, `miroir testsets
    # commons-gallery --gallery full`), merged into the full-sample runs.
    "full-f2": IndexSpec(
        name="full-f2",
        a7_run="A7-embeddinggemma-2-512-full+F2",
        a3_run="A3-megaloc-322-full+F2",
        a7_query_run="A7-embeddinggemma-2-512-T2-T2b-T8c-tune-queries",
        a3_query_run="A3-megaloc-322-T2-T2b-T8c-tune-queries",
        saved_guess="guess-A7-embeddinggemma-2-512-full.npz",
    ),
    # full-f2 stored as all of POP would be (docs/results.md, "Storage for all of POP"):
    # MegaLoc at 1,024 dims on the GPU to shortlist, whole in int8 in host memory to score.
    "full-f2-shortlist": IndexSpec(
        name="full-f2-shortlist",
        a7_run="A7-embeddinggemma-2-512-full+F2",
        a3_run="A3-megaloc-322-full+F2",
        a7_query_run="A7-embeddinggemma-2-512-T2-T2b-T8c-tune-queries",
        a3_query_run="A3-megaloc-322-T2-T2b-T8c-tune-queries",
        saved_guess="guess-A7-embeddinggemma-2-512-full.npz",
        a3_shortlist_dims=1024,
    ),
}


# --- build -----------------------------------------------------------------------------------


def build_index(settings: Settings, name: str) -> Path:
    """Write the bundle `name` (see the module doc). Returns its folder."""
    spec = INDEXES.get(name)
    if spec is None:
        msg = f"unknown web index {name!r}; known: {', '.join(INDEXES)}"
        raise ValueError(msg)
    out = settings.web_index_dir(name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    a7 = read_vectors(settings.vectors_path(spec.a7_run))
    a3 = read_vectors(settings.vectors_path(spec.a3_run))
    ids = _shared_ids(a7, a3)
    _write_matrix(out / A7_FILE, a7, ids, settings.vectors_path(spec.a7_run))
    if spec.a3_shortlist_dims is None:
        _write_matrix(out / A3_FILE, a3, ids, settings.vectors_path(spec.a3_run))
    else:
        write_shortlist_store(out, a3, ids, spec.a3_shortlist_dims)
    (out / IDS_FILE).write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    if spec.a3_projection is not None:
        if spec.a3_shortlist_dims is not None:
            msg = f"{name}: a3_projection and a3_shortlist_dims cannot go together"
            raise ValueError(msg)
        shutil.copyfile(
            settings.vectors_path(spec.a3_run) / spec.a3_projection, out / A3_PROJECTION_FILE
        )

    records = read_records(settings.records_parquet, settings.images_parquet)
    data = catalog_data(records, _commons(settings, ids), ids)
    write_catalog(out / CATALOG_FILE, data)
    catalog = Catalog(data)
    logger.info(
        "catalog: %d images, %d records, %d entities",
        len(data.images),
        len(data.records),
        catalog.entity_count,
    )

    if settings.places_parquet.exists():  # step H2: `miroir geo places`
        add_places(settings, out, catalog)

    entity_of = {i.image_id: i.entity_id for i in data.images}
    if spec.saved_guess is not None:
        guess = read_guess(settings.index_dir / spec.saved_guess)
        logger.info("type guess read from %s", spec.saved_guess)
    else:
        guess = _train_guess(settings, a7, records, entity_of, ids)
    write_guess(out / GUESS_FILE, guess)

    vectors = _Vectors.load(out)
    queries = _QuerySets.load(settings, spec, ids, None)  # `_Vectors` projects them
    routing, cutoff_report = calibrate(catalog, vectors, guess, queries, name)
    write_routing(out / ROUTING_FILE, routing)
    report = {
        "index": name,
        "cutoffs": cutoff_report,
        "ranking": score_sets(catalog, vectors, guess, routing, queries),
    }
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    saved_sure = settings.index_dir / f"sure-{name}.json"
    if saved_sure.exists():  # the object margin rule (`miroir web sure`) survives a rebuild
        shutil.copyfile(saved_sure, out / SURE_FILE)
    manifest = {
        "index": name,
        "a7_run": spec.a7_run,
        "a3_run": spec.a3_run,
        "a3_projection": spec.a3_projection,
        "a3_shortlist_dims": spec.a3_shortlist_dims,
        "images": len(ids),
        "records": len(data.records),
        "entities": catalog.entity_count,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def add_places(settings: Settings, folder: Path, catalog: Catalog | None = None) -> int:
    """Write `places.npz` into a bundle, for its catalog's entities. Returns how many are placed."""
    catalog = catalog or read_catalog(folder / CATALOG_FILE)
    placed = {p.entity_id: (p.latitude, p.longitude) for p in read_places(settings.places_parquet)}
    count = write_places(folder / PLACES_FILE, catalog.entity_ids, placed)
    logger.info("places: %d of %d entities placed", count, catalog.entity_count)
    return count


def catalog_data(
    records: Sequence[Record], commons: Mapping[str, CatalogImage], ids: Sequence[str]
) -> CatalogData:
    """The catalog of the images `ids` (in that order) and the records they belong to."""
    image_of = {i.image_id: (r, i) for r in records for i in r.images}
    images: list[CatalogImage] = []
    kept: dict[str, Record] = {}
    missing: list[str] = []
    for image_id in ids:
        if image_id in commons:
            images.append(commons[image_id])
            continue
        found = image_of.get(image_id)
        if found is None:
            missing.append(image_id)
            continue
        record, image = found
        kept[record.ref] = record
        images.append(
            CatalogImage(
                image_id=image.image_id,
                entity_id=image.entity_id,
                ref=record.ref,
                url=image.url,
                image_type=image.image_type.value,
                is_cover=image.is_cover,
            )
        )
    if missing:
        msg = f"{len(missing)} vector id(s) are neither POP nor Commons images: {missing[:3]}"
        raise ValueError(msg)
    # An entity known here only by Commons photos still needs a record for its card.
    with_commons = {i.entity_id for i in images if i.source is Source.COMMONS}
    for record in records:
        if record.entity_id in with_commons and record.ref not in kept:
            kept[record.ref] = record
    return CatalogData(
        images=tuple(images),
        records=tuple(_catalog_record(r) for r in sorted(kept.values(), key=lambda r: r.ref)),
    )


def _catalog_record(record: Record) -> CatalogRecord:
    return CatalogRecord(
        ref=record.ref,
        collection=record.collection.value,
        entity_id=record.entity_id,
        title=record.title,
        description=record.description[:DESCRIPTION_CHARS],
        denomination=record.denomination,
        domain=record.domain,
        authors=record.authors,
        period=record.period,
        region=record.region,
        departement=record.departement,
        commune=record.commune,
        pop_url=record.pop_url,
    )


def _commons(settings: Settings, ids: Sequence[str]) -> dict[str, CatalogImage]:
    """Commons photos among the index ids: extra views of an entity, with their credit."""
    wanted = {i for i in ids if i.startswith("commons-")}
    if not wanted:
        return {}
    out: dict[str, CatalogImage] = {}
    tables = [settings.commons_parquet, *settings.commons_gallery_parquets()]
    photos = [p for t in tables if t.exists() for p in read_commons(t)]
    for photo in photos:
        if photo.image_id in wanted and photo.image_id not in out:
            out[photo.image_id] = CatalogImage(
                image_id=photo.image_id,
                entity_id=photo.entity_id,
                ref=None,
                url=photo.thumb_url or photo.url,
                image_type="main",
                is_cover=False,
                source=Source.COMMONS,
                licence=photo.licence,
                author=photo.author,
                page_url=photo.page_url,
            )
    return out


def _shared_ids(a7: VectorSet, a3: VectorSet) -> list[str]:
    in_a3 = set(a3.ids)
    ids = [i for i in a7.ids if i in in_a3]
    dropped = len(a7.ids) + len(a3.ids) - 2 * len(ids)
    if dropped:
        logger.warning("%d image(s) have a vector from one model only: left out", dropped)
    return ids


def _write_matrix(path: Path, run: VectorSet, ids: Sequence[str], folder: Path) -> None:
    if tuple(ids) == run.ids:
        shutil.copyfile(folder / "vectors.npy", path)
        return
    rows = _rows_in_order(run, ids)
    np.save(path, np.asarray(run.vectors[rows], dtype=np.float16), allow_pickle=False)


def write_shortlist_store(folder: Path, run: VectorSet, ids: Sequence[str], dims: int) -> None:
    """MegaLoc as a shortlist bundle keeps it: PCA-reduced float16 rows (`a3.npy`) and their
    projection, and every row whole in int8 (`miroir.web.store`). The PCA is fitted on the
    run's own rows (gallery images, never queries)."""
    from miroir.index.pca import fit_pca  # noqa: PLC0415 (only shortlist bundles need it)

    pca = fit_pca(run, dims, run.info.option)
    rows = _rows_in_order(run, ids)
    mean = pca.mean.astype(np.float32)
    reduced = np.empty((len(rows), dims), dtype=np.float16)
    for start in range(0, len(rows), GALLERY_CHUNK):
        block = np.asarray(run.vectors[rows[start : start + GALLERY_CHUNK]], dtype=np.float32)
        projected = (block - mean) @ pca.components
        reduced[start : start + len(block)] = projected / np.linalg.norm(
            projected, axis=1, keepdims=True
        )
    np.save(folder / A3_FILE, reduced, allow_pickle=False)
    np.savez(folder / A3_PROJECTION_FILE, mean=mean, components=pca.components.T.copy())
    write_exact(folder, run.vectors, rows)
    logger.info(
        "MegaLoc shortlist store: %d dims (%.0f%% of variance) on the GPU, int8 in memory",
        dims,
        100 * pca.explained.sum(),
    )


def _rows_in_order(run: VectorSet, ids: Sequence[str]) -> NDArray[np.int64]:
    row_of = run.index()
    return np.fromiter((row_of[i] for i in ids), dtype=np.int64, count=len(ids))


def _train_guess(
    settings: Settings,
    a7: VectorSet,
    records: Sequence[Record],
    entity_of: Mapping[str, str],
    ids: Sequence[str],
) -> Guess:
    """G4's guess, trained as `miroir eval route` trains it: tune-query entities left out."""
    kinds: dict[str, router.PhotoType] = {}
    for record in records:
        kind = router.photo_type(record)
        if kind is not None:
            kinds.update((i.image_id, kind) for i in record.images)
    left_out = {
        e
        for s in router.QUERY_SETS
        for q in load_queries(settings, s, "tune")
        for e in (q.entity_id, *q.expected_entity_ids)
    }
    labels = {i: kinds[i] for i in ids if i in kinds and entity_of[i] not in left_out}
    fitted, held_out = router.train_guess([a7], labels, dict(entity_of))
    logger.info(
        "type guess: %d training images, held-out balanced accuracy %.3f", len(labels), held_out
    )
    return Guess(mean=fitted.mean, scale=fitted.scale, coef=fitted.coef, bias=fitted.bias)


# --- scoring the site's ranking on the test sets --------------------------------------------


@dataclass(frozen=True, slots=True)
class _Vectors:
    ids: tuple[str, ...]
    a7: NDArray[np.float16]
    a3: NDArray[np.float16]
    """MegaLoc as the GPU holds it: whole, or reduced when `projection` is set."""
    projection: Projection | None = None
    exact: ExactStore | None = None
    """Shortlist bundles: every MegaLoc vector whole, in int8 (`miroir.web.store`)."""

    @classmethod
    def load(cls, folder: Path) -> _Vectors:
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
class _Asked:
    """Queries of one set: their vectors, what they may not find, what counts as right."""

    set_id: str
    query_ids: tuple[str, ...]
    a7: Floats
    a3: Floats
    hidden: tuple[tuple[str, ...], ...]
    right: tuple[tuple[str, ...], ...]
    chroma: tuple[float, ...]
    """Each query photo's chroma spread (`images.colour`), as the server measures it."""


@dataclass(frozen=True, slots=True)
class _QuerySets:
    sets: dict[str, _Asked]

    @classmethod
    def load(
        cls, settings: Settings, spec: IndexSpec, ids: Sequence[str], projection: Projection | None
    ) -> _QuerySets:
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
) -> _Asked:
    row7, row3 = a7.index(), a3.index()
    v7 = np.asarray(a7.vectors[[row7[q.query_image_id] for q in queries]], dtype=np.float32)
    v3 = np.asarray(a3.vectors[[row3[q.query_image_id] for q in queries]], dtype=np.float32)
    if projection is not None:
        v3 = np.stack([projection.apply(v) for v in v3]) if len(v3) else v3
    return _Asked(
        set_id=set_id,
        query_ids=tuple(q.query_id for q in queries),
        a7=v7,
        a3=v3,
        hidden=tuple((q.query_image_id, *q.exclude_image_ids) for q in queries),
        right=tuple(q.expected_entity_ids for q in queries),
        chroma=tuple(chroma[q.query_image_id] for q in queries),
    )


@dataclass(frozen=True, slots=True)
class _Searched:
    """Per query: rank of the first right entity (0 when none is in the index), the top
    entity's confidence, and the bucket the query went to."""

    ranks: NDArray[np.int64]
    confidence: NDArray[np.float64]
    buckets: tuple[str, ...]


def _search_set(
    catalog: Catalog, vectors: _Vectors, guess: Guess, routing: Routing, asked: _Asked
) -> _Searched:
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
        ranks.append(right_rank(fused, right))
    return _Searched(np.asarray(ranks), np.asarray(confidence), tuple(buckets))


def right_rank(fused: Floats, right: Sequence[int]) -> int:
    """Rank of the first right entity (rows `right`, each with an image left to find); 0 when
    there is none. One the ranking leaves out (-inf: off a MegaLoc shortlist) is a miss, ranked
    past every entity, not a query skipped."""
    if not right:
        return 0
    order = ranking.top_entities(fused, len(fused))
    position = np.flatnonzero(np.isin(order, right))
    return int(position[0]) + 1 if position.size else len(fused) + 1


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


def calibrate(
    catalog: Catalog, vectors: _Vectors, guess: Guess, queries: _QuerySets, index: str
) -> tuple[Routing, dict[str, object]]:
    """Weights from R2's G5; one cut-off per bucket from T8 (at most WRONG_TARGET confident).

    A bucket that T8 barely reaches (colour photos of buildings) takes its cut-off from T8c,
    Commons photos with no record, against T2 and T2b as the searches that should match.
    """
    draft = Routing(
        buckets=(
            Bucket(
                "archive-building", ARCHIVE_A7_WEIGHT, math.inf, PhotoType.BUILDING, greyscale=True
            ),
            Bucket(
                "colour-building",
                COLOUR_BUILDING_A7_WEIGHT,
                math.inf,
                PhotoType.BUILDING,
                greyscale=False,
            ),
            Bucket("object", OBJECT_A7_WEIGHT, math.inf, PhotoType.OBJECT),
        )
    )

    def searched(set_ids: Sequence[str]) -> list[_Searched]:
        return [
            _search_set(catalog, vectors, guess, draft, queries.sets[s])
            for s in set_ids
            if s in queries.sets
        ]

    pools = {
        "T8": (searched(["T8"]), searched(["T1", "T5"]), "T1+T5"),
        "T8c": (searched(["T8c"]), searched(["T2", "T2b"]), "T2+T2b"),
    }
    date = datetime.now(UTC).date().isoformat()
    buckets: list[Bucket] = []
    report: dict[str, object] = {}
    for draft_bucket in draft.buckets:
        source = "T8"
        lost_runs, found_runs, found_sets = pools[source]
        lost = _in_bucket(lost_runs, draft_bucket.name)[0]
        if lost.size < MIN_CALIBRATION_QUERIES and pools["T8c"][0]:
            source = "T8c"
            lost_runs, found_runs, found_sets = pools[source]
            lost = _in_bucket(lost_runs, draft_bucket.name)[0]
        found_conf, found_hit1 = _in_bucket(found_runs, draft_bucket.name)
        cutoff = cutoff_for(lost, WRONG_TARGET)
        matched = found_conf >= cutoff
        stats = {
            "cutoff": cutoff,
            "calibrated_on": source,
            "not_found_queries": int(lost.size),
            "not_found_confident": float((lost >= cutoff).mean()) if lost.size else 0.0,
            "in_index_sets": found_sets,
            "in_index_queries": int(found_conf.size),
            "in_index_matched": float(matched.mean()) if matched.size else 0.0,
            "matched_right_at_1": float(found_hit1[matched].mean()) if matched.any() else 0.0,
        }
        report[draft_bucket.name] = stats
        note = (
            f"A7 weight {draft_bucket.a7_weight} from R2 G5 with F2 (docs/results.md). Cut-off: "
            f"lowest confidence keeping at most {WRONG_TARGET:.0%} of {source} tune searches "
            f"confident ({stats['not_found_queries']} {draft_bucket.name} queries, index "
            f"{index}, {date}); it lets {stats['in_index_matched']:.0%} of "
            f"{stats['in_index_queries']} {found_sets} tune searches match, "
            f"{stats['matched_right_at_1']:.0%} of those right at rank 1"
        )
        buckets.append(replace(draft_bucket, cutoff=cutoff, note=note))
    routing = Routing(
        buckets=tuple(buckets),
        source=f"miroir web build-index {index}, {date}",
    )
    return routing, report


def _in_bucket(
    runs: Sequence[_Searched], bucket: str
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Confidences of the searches that went to `bucket`, and whether each was right at 1."""
    conf = [r.confidence[np.asarray(r.buckets) == bucket] for r in runs]
    hit1 = [(r.ranks == 1)[np.asarray(r.buckets) == bucket] for r in runs]
    if not conf:
        return np.zeros(0), np.zeros(0, dtype=bool)
    return np.concatenate(conf), np.concatenate(hit1)


def cutoff_for(confidences: NDArray[np.float64], wrong_target: float) -> float:
    """Lowest cut-off at which at most `wrong_target` of these (wrong) searches are confident.

    A search is confident when its confidence is at or above the cut-off.
    """
    if confidences.size == 0:
        msg = "no not-found queries to set a cut-off from"
        raise ValueError(msg)
    allowed = math.floor(wrong_target * confidences.size)
    ordered = np.sort(confidences)[::-1]
    return float(np.nextafter(ordered[allowed], np.inf))


def score_sets(
    catalog: Catalog, vectors: _Vectors, guess: Guess, routing: Routing, queries: _QuerySets
) -> dict[str, dict[str, float]]:
    """Hit rates of the site's ranking on each set's tune half, and "not found" rates."""
    out: dict[str, dict[str, float]] = {}
    cutoff = {b.name: b.cutoff for b in routing.buckets}
    for set_id, asked in queries.sets.items():
        if not asked.query_ids:
            continue
        searched = _search_set(catalog, vectors, guess, routing, asked)
        confident = searched.confidence >= np.asarray([cutoff[b] for b in searched.buckets])
        row: dict[str, float] = {"queries": float(len(asked.query_ids))}
        if set_id.startswith("T8"):
            row["confident_but_wrong"] = float(confident.mean())
        else:
            scored = searched.ranks > 0
            ranks = searched.ranks[scored]
            row["scored"] = float(scored.sum())
            for k in (1, 3, 5, 10):
                row[f"hit@{k}"] = float(((ranks >= 1) & (ranks <= k)).mean())
            row["matched"] = float(confident[scored].mean())
            row["matched_right_at_1"] = (
                float((ranks[confident[scored]] == 1).mean()) if confident[scored].any() else 0.0
            )
        out[set_id] = row
        logger.info("%s: %s", set_id, ", ".join(f"{k} {v:.3f}" for k, v in row.items()))
    return out


# --- upload ------------------------------------------------------------------------------------


def bundle_files(folder: Path) -> list[Path]:
    if not (folder / MANIFEST_FILE).exists():
        msg = f"{folder} is not a finished bundle (no {MANIFEST_FILE}): run build-index"
        raise FileNotFoundError(msg)
    return sorted(p for p in folder.iterdir() if p.is_file())


# --- the site's search on stored or new vectors, for step C and E1 measurements -------------


@dataclass(frozen=True, slots=True)
class SiteBundle:
    """A built bundle, read for measuring: the site's ranking without the models."""

    catalog: Catalog
    vectors: _Vectors
    guess: Guess
    routing: Routing

    @classmethod
    def load(cls, folder: Path) -> SiteBundle:
        return cls(
            catalog=read_catalog(folder / CATALOG_FILE),
            vectors=_Vectors.load(folder),
            guess=read_guess(folder / GUESS_FILE),
            routing=read_routing(folder / ROUTING_FILE),
        )


#: Queries searched at once by `site_search` (two float32 rows of every image each).
SEARCH_BATCH = 256


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
            wanted = [catalog.entity_row[e] for e in right[q] if e in catalog.entity_row]
            position = np.flatnonzero(np.isin(order, wanted)) if wanted else np.empty(0)
            out.append(
                SiteHit(
                    bucket=bucket.name,
                    photo_type=photo_type,
                    entities=tuple(catalog.entity_ids[int(e)] for e in best),
                    confidences=tuple(float(w * e7[e] + (1 - w) * e3[e]) for e in best),
                    best_images=tuple(images),
                    right_rank=int(position[0]) + 1 if position.size else 0,
                )
            )
    return out


def _image_cosines(queries: Floats, gallery: NDArray[np.float16]) -> Floats:
    sims = np.empty((len(queries), gallery.shape[0]), dtype=np.float32)
    for g in range(0, gallery.shape[0], GALLERY_CHUNK):
        block = np.asarray(gallery[g : g + GALLERY_CHUNK], dtype=np.float32)
        sims[:, g : g + len(block)] = queries @ block.T
    return sims


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


def asked_sets(settings: Settings, name: str) -> dict[str, AskedSet]:
    """The tune queries the bundle `name`'s report scores (T1, T3, T5, T8, T2, T2b, T8c)."""
    spec = INDEXES[name]
    folder = settings.web_index_dir(name)
    ids = (folder / IDS_FILE).read_text(encoding="utf-8").splitlines()
    loaded = _QuerySets.load(settings, spec, ids, read_projection(folder))
    out: dict[str, AskedSet] = {}
    for set_id, asked in loaded.sets.items():
        out[set_id] = AskedSet(
            set_id=set_id,
            query_ids=asked.query_ids,
            query_image_ids=tuple(h[0] for h in asked.hidden),
            a7=asked.a7,
            a3=asked.a3,
            hidden=asked.hidden,
            right=asked.right,
        )
    return out
