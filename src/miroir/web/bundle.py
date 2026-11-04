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

The cut-offs and the report come from `miroir.web.calibrate`; `miroir.web.measure` reads a
built bundle to measure it. Everything here is NumPy on stored vectors: no model runs.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.config import Settings
from miroir.eval import router
from miroir.eval.bm25 import DESCRIPTION_CHARS
from miroir.eval.vectors import VectorSet, read_vectors
from miroir.jobs.privacy import PublishableImages
from miroir.records.extra import read_site_records
from miroir.records.model import Record
from miroir.records.store import read_commons, read_places
from miroir.web.calibrate import calibrate, draft_routing, score_sets
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
from miroir.web.engine import A3_FILE, A3_PROJECTION_FILE, A7_FILE
from miroir.web.levels import LEVELS_FILE
from miroir.web.measure import GALLERY_CHUNK, IDS_FILE, BundleVectors, SiteBundle, tune_sets
from miroir.web.nearby import PLACES_FILE, write_places
from miroir.web.publishable import PUBLISHABLE_FILE, write_publishable
from miroir.web.routing import (
    GUESS_FILE,
    ROUTING_FILE,
    Guess,
    read_guess,
    write_guess,
    write_routing,
)
from miroir.web.store import write_exact
from miroir.web.sure import SURE_FILE

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
REPORT_FILE = "report.json"

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
    entity_records: bool = False
    """Keep every record of an entity the index shows, also those with no image row of their
    own (all of POP: an image two records share is one row, under one of them)."""


INDEXES: dict[str, IndexSpec] = {
    "screening-v2": IndexSpec(
        name="screening-v2",
        a7_run="A7-embeddinggemma-2-512-screening-v2-publishable",
        a3_run="A3-megaloc-322-screening-v2-publishable",
        a7_query_run="A7-embeddinggemma-2-512-T2-T2b-T8c-tune-queries",
        a3_query_run="A3-megaloc-322-T2-T2b-T8c-tune-queries",
    ),
    # Every image of the July exports the Modal runs could fetch (136,731 of 137,775), from
    # the full-sample index (then `embed modal --all-images`, now `embed scw`); MegaLoc kept
    # at 8,448 dims.
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
    # full-f2 plus the records copied from the whole download (`miroir records add`), so an
    # outside benchmark can find them: the Vétheuil Passion altarpiece (PM95000723) and its
    # 42 Mémoire photos, for the auction photos of its stolen panels.
    "full-f2-x": IndexSpec(
        name="full-f2-x",
        a7_run="A7-embeddinggemma-2-512-full+F2+X",
        a3_run="A3-megaloc-322-full+F2+X",
        a7_query_run="A7-embeddinggemma-2-512-T2-T2b-T8c-tune-queries",
        a3_query_run="A3-megaloc-322-T2-T2b-T8c-tune-queries",
        saved_guess="guess-A7-embeddinggemma-2-512-full.npz",
    ),
    # Every POP image (`miroir scw pop-download`, `pop-embed`, `pop-index`): the July vectors
    # and those made on Scaleway, MegaLoc stored as a shortlist, as `full-f2-shortlist`.
    "pop-all": IndexSpec(
        name="pop-all",
        a7_run="A7-embeddinggemma-2-512-pop-all",
        a3_run="A3-megaloc-322-pop-all",
        a7_query_run="A7-embeddinggemma-2-512-T2-T2b-T8c-tune-queries",
        a3_query_run="A3-megaloc-322-T2-T2b-T8c-tune-queries",
        saved_guess="guess-A7-embeddinggemma-2-512-full.npz",
        a3_shortlist_dims=1024,
        entity_records=True,
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

    records = read_site_records(settings)
    data = catalog_data(records, _commons(settings, ids), ids, entity_records=spec.entity_records)
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
    add_publishable(settings, out, ids)

    entity_of = {i.image_id: i.entity_id for i in data.images}
    if spec.saved_guess is not None:
        guess = read_guess(settings.index_dir / spec.saved_guess)
        logger.info("type guess read from %s", spec.saved_guess)
    else:
        guess = _train_guess(settings, a7, records, entity_of, ids)
    write_guess(out / GUESS_FILE, guess)

    vectors = BundleVectors.load(out)
    sets = tune_sets(settings, spec, catalog)
    unrouted = SiteBundle(catalog, vectors, guess, routing=draft_routing())
    routing, cutoff_report = calibrate(unrouted, sets, name)
    write_routing(out / ROUTING_FILE, routing)
    report = {
        "index": name,
        "cutoffs": cutoff_report,
        "ranking": score_sets(SiteBundle(catalog, vectors, guess, routing), sets),
    }
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    # The object margin rule (`miroir web sure`) and the card levels (`miroir web levels`)
    # survive a rebuild.
    for saved, target in ((f"sure-{name}.json", SURE_FILE), (f"levels-{name}.json", LEVELS_FILE)):
        if (settings.index_dir / saved).exists():
            shutil.copyfile(settings.index_dir / saved, out / target)
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


def add_publishable(settings: Settings, folder: Path, ids: Sequence[str] | None = None) -> int:
    """Write `publishable.txt.gz` into a bundle: its images that may go to an outside model
    service, by the rule of `PublishableImages`. Returns how many."""
    if ids is None:
        ids = (folder / IDS_FILE).read_text(encoding="utf-8").split()
    store = PublishableImages.load(settings)
    kept = store.only_publishable(ids)
    count = write_publishable(folder / PUBLISHABLE_FILE, kept)
    logger.info("publishable: %d of %d images may go to an outside service", count, len(ids))
    return count


def catalog_data(
    records: Sequence[Record],
    commons: Mapping[str, CatalogImage],
    ids: Sequence[str],
    *,
    entity_records: bool = False,
) -> CatalogData:
    """The catalog of the images `ids` (in that order) and the records they belong to; with
    `entity_records`, every record of their entities."""
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
    shown = (
        {i.entity_id for i in images}
        if entity_records
        else {i.entity_id for i in images if i.source is Source.COMMONS}
    )
    for record in records:
        if record.entity_id in shown and record.ref not in kept:
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
    """G4's guess, trained as every guess is (`router.guess_report`): every entity of every
    test set left out."""
    kinds: dict[str, router.PhotoType] = {}
    for record in records:
        kind = router.photo_type(record)
        if kind is not None:
            kinds.update((i.image_id, kind) for i in record.images)
    report = router.guess_report(settings, [a7], ids, entity_of, kinds)
    fitted = report.guess
    logger.info(
        "type guess: %d training images, held-out balanced accuracy %.3f",
        report.trained_on,
        report.held_out_accuracy,
    )
    return Guess(mean=fitted.mean, scale=fitted.scale, coef=fitted.coef, bias=fitted.bias)


# --- upload ------------------------------------------------------------------------------------


def bundle_files(folder: Path) -> list[Path]:
    if not (folder / MANIFEST_FILE).exists():
        msg = f"{folder} is not a finished bundle (no {MANIFEST_FILE}): run build-index"
        raise FileNotFoundError(msg)
    return sorted(p for p in folder.iterdir() if p.is_file())
