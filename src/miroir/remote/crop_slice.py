# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""The crop slice test on a Scaleway GPU machine (`miroir scw crop-slice-run`, in the job image).

Reads the spec the Mac wrote (`miroir.jobs.crop_slice`), downloads the `pop-all` bundle, and:

1. gets every query's vectors: stored ones for POP images (as `Engine.stored_cosines`),
   supplied ones from the Mac, or the site's models for outside photos (`prepare_photo`, as a
   visitor's photo); and its chroma spread and routed bucket, as the site does;
2. searches each query over the whole index as the site does (MegaLoc shortlist, exact
   rerank) and keeps a summary (`miroir.eval.crops.summarize_full`);
3. scores each query against the gallery rows' whole vectors, read from the bundle;
4. cuts every stored gallery file into the grids' windows, each shrunk like a whole photo
   (512 px JPEG, as `to_jpeg`), embeds them with MegaLoc (and EmbeddingGemma 2 for the sample
   `a7_files`) and keeps, per query, file and grid, the best crop cosine. Crop vectors are
   not kept.

Score tables go to the images bucket (`out_key`), the crop tables also every
`SAVE_EVERY_S` so a run cut short keeps what it did.
"""

from __future__ import annotations

import io
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import batched
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from miroir.config import MODAL_WEB_INDEX_MOUNT, MODAL_WEIGHTS_MOUNT, Settings
from miroir.eval.crops import FULL_TOP, summarize_full
from miroir.images.colour import chroma_spread
from miroir.images.fetch import JPEG_QUALITY, LONG_SIDE
from miroir.jobs.bulk import JobState, JobStatus, full_key, status_key
from miroir.jobs.crop_slice import (
    JOB,
    SPEC_KEY,
    SUPPLIED_KEY,
    QuerySource,
    SliceSpec,
    crop_boxes,
    out_key,
    outside_key,
    read_spec,
)
from miroir.jobs.modal_spec import OPTIONS
from miroir.jobs.objects import Bucket, bucket_name, download_tree, s3_client
from miroir.parallel import bounded_map
from miroir.remote.models import DEVICE, MegaLocEncoder, SentenceEncoder
from miroir.web import ranking
from miroir.web.catalog import CATALOG_FILE, Catalog, read_catalog
from miroir.web.engine import A3_FILE, A7_FILE, Projection, prepare_photo, read_projection, unit
from miroir.web.routing import GUESS_FILE, ROUTING_FILE, Guess, Routing, read_guess, read_routing
from miroir.web.store import ExactStore, read_exact, shortlist_cosines

logger = logging.getLogger(__name__)

INDEX = "pop-all"
A7 = OPTIONS["A7-embeddinggemma-2-512"]
A3 = OPTIONS["A3-megaloc-322"]
A3_BATCH = 64
A7_BATCH = 64
A7_STREAMS = 2
READERS = 12
#: Files cut and embedded together.
GROUP = 16
STATUS_EVERY_S = 30.0
SAVE_EVERY_S = 15 * 60.0


class Status:
    def __init__(self, s3, bucket: str) -> None:
        self.s3 = s3
        self.bucket = bucket
        self.status = JobStatus(JOB, 0, 1)
        self.counts: Counter[str] = Counter()
        self.reported = 0.0

    def set(self, *, force: bool = False, **changes: object) -> None:
        self.status = self.status.now(counts=dict(self.counts), **changes)
        if not force and time.time() - self.reported < STATUS_EVERY_S:
            return
        self.reported = time.time()
        body = self.status.to_json().encode()
        self.s3.put_object(Bucket=self.bucket, Key=status_key(JOB, 0), Body=body)
        logger.info("%s", self.status.line())


def _save(s3, bucket: str, name: str, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # pyright: ignore[reportArgumentType]
    s3.put_object(Bucket=bucket, Key=out_key(name), Body=buffer.getvalue())
    logger.info("saved %s (%.1f MB)", name, buffer.tell() / 1e6)


def _small(image: PILImage.Image) -> PILImage.Image:
    """A crop prepared as the whole photos were (`to_jpeg`: at most 512 px, LANCZOS, JPEG),
    then opened as RGB."""
    image.thumbnail((LONG_SIDE, LONG_SIDE), PILImage.Resampling.LANCZOS)
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=JPEG_QUALITY)
    with PILImage.open(io.BytesIO(out.getvalue())) as opened:
        return opened.convert("RGB")


def _crops(s3, bucket: str, file_id: str, spec: SliceSpec) -> list[PILImage.Image] | str:
    """Every window of every grid, in grid order, or why the file cannot be read."""
    try:
        data = s3.get_object(Bucket=bucket, Key=full_key(file_id))["Body"].read()
    except Exception as err:
        return f"read: {type(err).__name__}"
    try:
        with PILImage.open(io.BytesIO(data)) as opened:
            image = opened.convert("RGB")
        return [
            _small(image.crop(box))
            for grid in spec.grids
            for box in crop_boxes(image.width, image.height, grid)
        ]
    except Exception as err:
        return f"unreadable: {type(err).__name__}"


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        msg = "the model returned NaN or infinite values"
        raise FloatingPointError(msg)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class Models:
    def __init__(self) -> None:
        self.a7 = SentenceEncoder(replace(A7, batch_size=A7_BATCH))
        self.a3 = MegaLocEncoder(A3)
        self.pool = ThreadPoolExecutor(A7_STREAMS)

    def encode_a3(self, images: list[PILImage.Image]) -> np.ndarray:
        return _unit_rows(
            np.concatenate([self.a3.encode(list(b)) for b in batched(images, A3_BATCH)])
        )

    def encode_a7(self, images: list[PILImage.Image]) -> np.ndarray:
        parts = list(self.pool.map(lambda b: self.a7.encode(list(b)), batched(images, A7_BATCH)))
        return _unit_rows(np.concatenate(parts))


def run_crop_slice(settings: Settings) -> None:
    s3 = s3_client(settings)
    images_bucket = bucket_name(settings, Bucket.IMAGES)
    status = Status(s3, images_bucket)
    try:
        _run(settings, s3, images_bucket, status)
    except Exception:
        status.set(force=True, state=JobState.FAILED)
        raise
    status.set(force=True, state=JobState.DONE)


@dataclass(frozen=True, slots=True)
class Bundle:
    folder: Path
    catalog: Catalog
    routing: Routing
    guess: Guess
    projection: Projection | None
    exact: ExactStore
    a7: np.ndarray
    """EmbeddingGemma 2 vectors of every row, memory-mapped."""

    @classmethod
    def load(cls, folder: Path) -> Bundle:
        exact = read_exact(folder)
        if exact is None:
            msg = f"{folder}: pop-all keeps MegaLoc whole in int8"
            raise ValueError(msg)
        return cls(
            folder=folder,
            catalog=read_catalog(folder / CATALOG_FILE),
            routing=read_routing(folder / ROUTING_FILE),
            guess=read_guess(folder / GUESS_FILE),
            projection=read_projection(folder),
            exact=exact,
            a7=np.load(folder / A7_FILE, mmap_mode="r"),
        )

    def stored(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """One row's vectors, as `Engine.stored_cosines` reads them."""
        a7 = np.asarray(self.a7[row], dtype=np.float32)
        a3 = np.asarray(self.exact.rows[row], dtype=np.float32) * self.exact.scale[row]
        return unit(a7), unit(a3)


@dataclass(frozen=True, slots=True)
class Queries:
    a7: np.ndarray
    a3: np.ndarray
    weights: np.ndarray


def _run(settings: Settings, s3, bucket: str, status: Status) -> None:
    spec = read_spec(s3.get_object(Bucket=bucket, Key=SPEC_KEY)["Body"].read())
    status.set(force=True, note="weights and bundle")
    download_tree(s3, bucket_name(settings, Bucket.WEIGHTS), Path(MODAL_WEIGHTS_MOUNT))
    start = time.monotonic()
    root = Path(MODAL_WEB_INDEX_MOUNT)
    download_tree(s3, bucket_name(settings, Bucket.INDEX), root, prefix=f"{INDEX}/")
    logger.info("bundle down in %.0f s", time.monotonic() - start)
    status.set(force=True, note="loading")
    bundle = Bundle.load(root / INDEX)
    models = Models()
    queries = _queries(s3, bucket, spec, bundle, models)
    status.set(force=True, note="full index")
    _full(s3, bucket, spec, bundle, queries=queries, status=status)
    status.set(force=True, note="gallery whole")
    _whole(s3, bucket, spec, bundle, queries)
    _all_crops(s3, bucket, spec, models, queries=queries, status=status)


def _body(s3, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _queries(s3, bucket: str, spec: SliceSpec, bundle: Bundle, models: Models) -> Queries:
    """Every query's unit vectors, chroma spread and routed weight (saved as "queries")."""
    supplied: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    with np.load(io.BytesIO(_body(s3, bucket, SUPPLIED_KEY))) as data:
        for i, image_id in enumerate(data["ids"].tolist()):
            supplied[str(image_id)] = (data["a7"][i], data["a3"][i], float(data["chroma"][i]))
    q7: list[np.ndarray] = []
    q3: list[np.ndarray] = []
    chroma: list[float] = []
    for query in spec.queries:
        if query.source is QuerySource.CATALOG:
            v7, v3 = bundle.stored(bundle.catalog.image_row[query.image_id])
            c = chroma_spread(prepare_photo(_body(s3, bucket, full_key(query.file_id))))
        elif query.source is QuerySource.SUPPLIED:
            v7, v3, c = supplied[query.image_id]
        else:
            jpeg = prepare_photo(_body(s3, bucket, outside_key(query.image_id)))
            with PILImage.open(io.BytesIO(jpeg)) as opened:
                picture = opened.convert("RGB")
            v7, v3 = models.a7.encode([picture])[0], models.a3.encode([picture])[0]
            c = chroma_spread(jpeg)
        q7.append(unit(v7))
        q3.append(unit(v3))
        chroma.append(c)
    a7, a3 = np.stack(q7), np.stack(q3)
    buckets = [
        bundle.routing.choose(bundle.guess.photo_type(a7[j])[0], chroma[j]) for j in range(len(a7))
    ]
    weights = np.asarray([b.a7_weight for b in buckets], dtype=np.float32)
    _save(
        s3, bucket, "queries",
        query_ids=np.array([q.query_id for q in spec.queries]),
        a7=a7, chroma=np.asarray(chroma, dtype=np.float32), a7_weight=weights,
        buckets=np.array([b.name for b in buckets]),
    )  # fmt: skip
    return Queries(a7, a3, weights)


def _gpu(matrix: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(matrix, dtype=np.float16)).to(DEVICE)


@torch.inference_mode()
def _product(matrix: torch.Tensor, vectors: np.ndarray) -> np.ndarray:
    """`matrix` (on the GPU) times each of `vectors`, in float32 on the CPU."""
    return (matrix @ _gpu(vectors).T).float().cpu().numpy()


def _gallery_entities(spec: SliceSpec, catalog: Catalog) -> tuple[np.ndarray, list[str]]:
    rows = np.asarray([catalog.image_row[i] for i in spec.gallery], dtype=np.int64)
    entities = sorted({catalog.entity_ids[int(catalog.image_entity[r])] for r in rows})
    return rows, entities


def _full(
    s3, bucket: str, spec: SliceSpec, bundle: Bundle, *, queries: Queries, status: Status
) -> None:
    """Each query over the whole index, as the site searches it (saved as "full")."""
    catalog = bundle.catalog
    _, entities = _gallery_entities(spec, catalog)
    entity_rows = np.asarray([catalog.entity_row[e] for e in entities], dtype=np.int64)
    g7 = _gpu(np.load(bundle.folder / A7_FILE))
    g3 = _gpu(np.load(bundle.folder / A3_FILE))
    fields: dict[str, list[np.ndarray]] = {
        k: [] for k in ("top", "top_e7", "top_e3", "gallery_e7", "gallery_e3", "range")
    }
    for j, query in enumerate(spec.queries):
        a3 = queries.a3[j]
        reduced = a3 if bundle.projection is None else bundle.projection.apply(a3)
        s7 = _product(g7, queries.a7[j : j + 1])[:, 0]
        s3r = _product(g3, reduced[None, :])[:, 0]
        hidden = [catalog.image_row[i] for i in query.hidden if i in catalog.image_row]
        s7[hidden] = -np.inf
        s3r[hidden] = -np.inf
        exact = shortlist_cosines(bundle.exact, a3, s7, s3r)
        e7 = ranking.entity_max(s7, catalog)
        e3 = ranking.entity_max(exact, catalog)
        full = summarize_full(e7, e3, float(queries.weights[j]), entity_rows, FULL_TOP)
        fields["top"].append(full.top)
        fields["top_e7"].append(full.top_e7)
        fields["top_e3"].append(full.top_e3)
        fields["gallery_e7"].append(full.gallery_e7)
        fields["gallery_e3"].append(full.gallery_e3)
        fields["range"].append(np.asarray([full.lo7, full.hi7, full.lo3, full.hi3]))
        status.set(note=f"full index: {j + 1}/{len(spec.queries)} queries")
    del g7, g3
    torch.cuda.empty_cache()
    named = sorted({int(r) for top in fields["top"] for r in top})
    _save(
        s3, bucket, "full",
        top=np.stack(fields["top"]),
        top_e7=np.stack(fields["top_e7"]).astype(np.float32),
        top_e3=np.stack(fields["top_e3"]).astype(np.float32),
        gallery_e7=np.stack(fields["gallery_e7"]).astype(np.float32),
        gallery_e3=np.stack(fields["gallery_e3"]).astype(np.float32),
        range=np.stack(fields["range"]),
        named_rows=np.asarray(named, dtype=np.int64),
        named_entities=np.array([catalog.entity_ids[r] for r in named]),
        gallery_entities=np.array(entities),
    )  # fmt: skip


def _whole(s3, bucket: str, spec: SliceSpec, bundle: Bundle, queries: Queries) -> None:
    """Each query against each gallery row's stored vectors (saved as "whole")."""
    catalog, exact = bundle.catalog, bundle.exact
    rows, _ = _gallery_entities(spec, catalog)
    g7 = _gpu(bundle.a7[rows])
    g3 = _gpu(np.asarray(exact.rows[rows], dtype=np.float32) * exact.scale[rows][:, None])
    whole7 = _product(g7, queries.a7).T.astype(np.float16)
    whole3 = _product(g3, queries.a3).T.astype(np.float16)
    del g7, g3
    torch.cuda.empty_cache()
    _save(
        s3, bucket, "whole",
        image_ids=np.array(spec.gallery), files=np.array(spec.files),
        entities=np.array([catalog.entity_ids[int(catalog.image_entity[r])] for r in rows]),
        whole7=whole7, whole3=whole3,
    )  # fmt: skip


class CropTables:
    """Per query, file and grid: the best crop cosine, per model (-inf until done)."""

    def __init__(self, spec: SliceSpec, files: list[str], n_queries: int) -> None:
        self.spec = spec
        self.files = files
        self.a7_files = set(spec.a7_files)
        self.counts = [len(crop_boxes(100, 100, g)) for g in spec.grids]
        shape = (n_queries, len(files))
        self.tables = {
            (model, g.name): np.full(shape, -np.inf, dtype=np.float16)
            for model in ("a3", "a7")
            for g in spec.grids
        }
        self.best = {g.name: np.full(shape, -1, dtype=np.int8) for g in spec.grids}
        """Which window of the grid scored best with MegaLoc."""
        self.done = np.zeros(len(files), dtype=bool)

    def put(self, model: str, cols: list[int], sims: np.ndarray) -> None:
        """`sims`: (files, windows, queries), windows in grid order."""
        at = 0
        for grid, count in zip(self.spec.grids, self.counts, strict=True):
            part = sims[:, at : at + count, :]
            self.tables[(model, grid.name)][:, cols] = part.max(axis=1).T
            if model == "a3":
                self.best[grid.name][:, cols] = part.argmax(axis=1).T
            at += count

    def save(self, s3, bucket: str) -> None:
        _save(
            s3, bucket, "crops",
            files=np.array(self.files), done=self.done,
            a7_files=np.array([f in self.a7_files for f in self.files]),
            **{f"{m}_{g}": t for (m, g), t in self.tables.items()},
            **{f"a3_{g}_best": b for g, b in self.best.items()},
        )  # fmt: skip


def _all_crops(
    s3, bucket: str, spec: SliceSpec, models: Models, *, queries: Queries, status: Status
) -> None:
    """Cut, embed and score every stored gallery file (saved as "crops")."""
    files = spec.crop_files()
    tables = CropTables(spec, files, len(queries.a7))
    gpu = {"a3": _gpu(queries.a3), "a7": _gpu(queries.a7)}
    status.status = replace(status.status, total=len(files), done=0)
    saved = started = time.monotonic()
    embedded = 0

    def load(group: tuple[int, ...]) -> list[tuple[int, list[PILImage.Image] | str]]:
        return [(c, _crops(s3, bucket, files[c], spec)) for c in group]

    groups = batched(range(len(files)), GROUP)
    for group in bounded_map(load, groups, workers=READERS, window=READERS):
        kept: list[tuple[int, list[PILImage.Image]]] = []
        for c, crops in group:
            if isinstance(crops, str):
                status.counts[crops.split(":")[0]] += 1
            else:
                status.counts["ok"] += 1
                kept.append((c, crops))
        for model in ("a3", "a7"):
            wanted = kept if model == "a3" else [k for k in kept if files[k[0]] in tables.a7_files]
            if not wanted:
                continue
            shots = [p for _, crops in wanted for p in crops]
            vectors = models.encode_a3(shots) if model == "a3" else models.encode_a7(shots)
            sims = _product(gpu[model], vectors).T  # (shots, queries)
            per_file = len(wanted[0][1])
            tables.put(model, [c for c, _ in wanted], sims.reshape(len(wanted), per_file, -1))
            embedded += len(shots)
        tables.done[[c for c, _ in group]] = True
        elapsed = time.monotonic() - started
        status.set(
            done=int(tables.done.sum()),
            note=f"crops: {embedded:,} in {elapsed:.0f} s ({embedded / max(elapsed, 1):.1f}/s)",
        )
        if time.monotonic() - saved > SAVE_EVERY_S:
            tables.save(s3, bucket)
            saved = time.monotonic()
    tables.save(s3, bucket)
