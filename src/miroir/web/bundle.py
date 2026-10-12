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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.config import Settings
from miroir.eval import router
from miroir.eval.sets import Query, load_queries
from miroir.eval.vectors import VectorSet, read_vectors
from miroir.records.model import Record
from miroir.records.store import read_commons, read_records
from miroir.web import ranking
from miroir.web.catalog import (
    CATALOG_FILE,
    Catalog,
    CatalogData,
    CatalogImage,
    CatalogRecord,
    Source,
    write_catalog,
)
from miroir.web.engine import A3_FILE, A3_PROJECTION_FILE, A7_FILE, Projection, read_projection
from miroir.web.routing import (
    GUESS_FILE,
    ROUTING_FILE,
    Bucket,
    Guess,
    PhotoType,
    Routing,
    read_guess,
    write_guess,
    write_routing,
)

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

#: G4 weights from R2 (docs/results.md, "G4"): the grid picked EmbeddingGemma 2 at 1.0 for
#: buildings and 0.2 for objects (blend), mean hit@10 67.6 on T1, T3 and T5 tune.
BUILDING_A7_WEIGHT = 1.0
OBJECT_A7_WEIGHT = 0.2

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
    _write_matrix(out / A3_FILE, a3, ids, settings.vectors_path(spec.a3_run))
    (out / IDS_FILE).write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    if spec.a3_projection is not None:
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

    entity_of = {i.image_id: i.entity_id for i in data.images}
    if spec.saved_guess is not None:
        guess = read_guess(settings.index_dir / spec.saved_guess)
        logger.info("type guess read from %s", spec.saved_guess)
    else:
        guess = _train_guess(settings, a7, records, entity_of, ids)
    write_guess(out / GUESS_FILE, guess)

    vectors = _Vectors.load(out)
    queries = _QuerySets.load(settings, spec, ids, read_projection(out))
    routing, cutoff_report = calibrate(catalog, vectors, guess, queries, name)
    write_routing(out / ROUTING_FILE, routing)
    report = {
        "index": name,
        "cutoffs": cutoff_report,
        "ranking": score_sets(catalog, vectors, guess, routing, queries),
    }
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "index": name,
        "a7_run": spec.a7_run,
        "a3_run": spec.a3_run,
        "a3_projection": spec.a3_projection,
        "images": len(ids),
        "records": len(data.records),
        "entities": catalog.entity_count,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return out


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
    for photo in read_commons(settings.commons_parquet):
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
    row_of = run.index()
    rows = np.fromiter((row_of[i] for i in ids), dtype=np.int64, count=len(ids))
    np.save(path, np.asarray(run.vectors[rows], dtype=np.float16), allow_pickle=False)


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

    @classmethod
    def load(cls, folder: Path) -> _Vectors:
        ids = tuple((folder / IDS_FILE).read_text(encoding="utf-8").splitlines())
        return cls(
            ids=ids,
            a7=np.load(folder / A7_FILE, mmap_mode="r"),
            a3=np.load(folder / A3_FILE, mmap_mode="r"),
        )


@dataclass(frozen=True, slots=True)
class _Asked:
    """Queries of one set: their vectors, what they may not find, what counts as right."""

    set_id: str
    query_ids: tuple[str, ...]
    a7: Floats
    a3: Floats
    hidden: tuple[tuple[str, ...], ...]
    right: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class _QuerySets:
    sets: dict[str, _Asked]

    @classmethod
    def load(
        cls, settings: Settings, spec: IndexSpec, ids: Sequence[str], projection: Projection | None
    ) -> _QuerySets:
        index = set(ids)
        gallery_a7 = read_vectors(settings.vectors_path(spec.a7_run))
        gallery_a3 = read_vectors(settings.vectors_path(spec.a3_run))
        out: dict[str, _Asked] = {}
        for set_id in ("T1", "T3", "T5", "T8"):
            queries = [
                q for q in load_queries(settings, set_id, "tune") if q.query_image_id in index
            ]
            out[set_id] = _asked(set_id, queries, gallery_a7, gallery_a3, None)
        if spec.a7_query_run and spec.a3_query_run:
            q7 = read_vectors(settings.vectors_path(spec.a7_query_run))
            q3 = read_vectors(settings.vectors_path(spec.a3_query_run))
            have = set(q7.ids) & set(q3.ids)
            for set_id in ("T2", "T2b", "T8c"):
                queries = [
                    q for q in load_queries(settings, set_id, "tune") if q.query_image_id in have
                ]
                out[set_id] = _asked(set_id, queries, q7, q3, projection)
        return cls(out)


def _asked(
    set_id: str,
    queries: Sequence[Query],
    a7: VectorSet,
    a3: VectorSet,
    projection: Projection | None,
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
    e3 = entity_cosines(asked.a3, vectors.a3, catalog, asked.hidden)
    ranks: list[int] = []
    confidence: list[float] = []
    buckets: list[str] = []
    for j in range(len(asked.query_ids)):
        photo_type, _ = guess.photo_type(asked.a7[j])
        bucket = routing.choose(photo_type, None)
        w = bucket.a7_weight
        signals = [
            s for s in (ranking.Signal(e7[j], w), ranking.Signal(e3[j], 1 - w)) if s.weight > 0
        ]
        fused = ranking.fuse(signals)
        top = int(np.argmax(np.where(np.isfinite(fused), fused, -np.inf)))
        confidence.append(float(w * e7[j, top] + (1 - w) * e3[j, top]))
        buckets.append(bucket.name)
        right = [catalog.entity_row[e] for e in asked.right[j] if e in catalog.entity_row]
        if not right:
            ranks.append(0)
            continue
        order = ranking.top_entities(fused, catalog.entity_count)
        position = np.flatnonzero(np.isin(order, right))
        ranks.append(int(position[0]) + 1 if position.size else 0)
    return _Searched(np.asarray(ranks), np.asarray(confidence), tuple(buckets))


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
    """Weights from R2's G4; one cut-off per bucket from T8 (at most WRONG_TARGET confident)."""
    draft = Routing(
        buckets=(
            Bucket("building", BUILDING_A7_WEIGHT, math.inf, PhotoType.BUILDING),
            Bucket("object", OBJECT_A7_WEIGHT, math.inf, PhotoType.OBJECT),
        )
    )
    not_found = _search_set(catalog, vectors, guess, draft, queries.sets["T8"])
    in_index = [_search_set(catalog, vectors, guess, draft, queries.sets[s]) for s in ("T1", "T5")]
    found_conf = np.concatenate([s.confidence for s in in_index])
    found_buckets = tuple(b for s in in_index for b in s.buckets)
    found_hit1 = np.concatenate([s.ranks == 1 for s in in_index])
    date = datetime.now(UTC).date().isoformat()
    buckets: list[Bucket] = []
    report: dict[str, object] = {}
    for draft_bucket in draft.buckets:
        lost = not_found.confidence[np.asarray(not_found.buckets) == draft_bucket.name]
        mine = np.asarray(found_buckets) == draft_bucket.name
        cutoff = cutoff_for(lost, WRONG_TARGET)
        matched = found_conf[mine] >= cutoff
        stats = {
            "cutoff": cutoff,
            "t8_queries": int(lost.size),
            "t8_confident": float((lost >= cutoff).mean()) if lost.size else 0.0,
            "in_index_queries": int(mine.sum()),
            "in_index_matched": float(matched.mean()) if matched.size else 0.0,
            "matched_right_at_1": float(found_hit1[mine][matched].mean()) if matched.any() else 0.0,
        }
        report[draft_bucket.name] = stats
        note = (
            f"A7 weight {draft_bucket.a7_weight} from R2 G4 (docs/results.md). Cut-off: lowest "
            f"confidence keeping at most {WRONG_TARGET:.0%} of T8 tune searches confident "
            f"({stats['t8_queries']} {draft_bucket.name} queries, index {index}, {date}); "
            f"it lets {stats['in_index_matched']:.0%} of {stats['in_index_queries']} T1+T5 "
            f"tune searches match, {stats['matched_right_at_1']:.0%} of those right at rank 1"
        )
        buckets.append(
            Bucket(
                draft_bucket.name,
                draft_bucket.a7_weight,
                cutoff,
                draft_bucket.photo_type,
                note=note,
            )
        )
    routing = Routing(
        buckets=tuple(buckets),
        source=f"miroir web build-index {index}, {date}",
    )
    return routing, report


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
            for k in (1, 5, 10):
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
