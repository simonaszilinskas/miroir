# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Crop vectors on a Scaleway GPU machine (`miroir scw crop-embed-run`, in the job image).

Reads the `CropSpec` the Mac wrote (`miroir.jobs.crop_embed`) and does pieces `first` to
`last` that are not written yet. Each stored file (at most 1,024 px) is cut into the grid's
windows; each window is shrunk as the whole photos were (`as_gallery_crop`: 512 px JPEG), then
resized to MegaLoc's 322 x 322 as `miroir.remote.preprocess.squash` does, on the processor's
threads; the GPU normalises and embeds them. A piece is written once all its files are done:
whole vectors (the slice) or projected ones (all of POP).

Before the run, `CropSpec.trial` crops are embedded in every precision and compared with
float32 (speed and cosines, `trial_key`). The slice also writes its queries' MegaLoc vectors
(`queries_key`): stored rows read from the `pop-all` bundle's int8 store, the Mac's supplied
vectors, or MegaLoc in float32 for outside photos, as the site embeds a visitor's photo.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from miroir.config import WEIGHTS_MOUNT, Settings, get_settings
from miroir.jobs.bulk import FULL_PREFIX, JobState, JobStatus, full_key, status_key
from miroir.jobs.crop_embed import (
    CropSpec,
    Form,
    Precision,
    job_name,
    piece_key,
    queries_key,
    read_crop_spec,
    spec_key,
    trial_key,
    whole_key,
    write_piece,
    written_pieces,
)
from miroir.jobs.crop_slice import SUPPLIED_KEY, Grid, QuerySource, crop_boxes, outside_key
from miroir.jobs.objects import Bucket, bucket_name, download_tree, listing, s3_client
from miroir.jobs.remote_spec import OPTIONS
from miroir.parallel import bounded_map, process_map
from miroir.remote.crop_slice import as_gallery_crop
from miroir.remote.models import DEVICE
from miroir.remote.preprocess import IMAGENET_MEAN, IMAGENET_STD, squash
from miroir.web.engine import prepare_photo
from miroir.web.store import quantize_rows

logger = logging.getLogger(__name__)

A3 = OPTIONS["A3-megaloc-322"]
SIDE = A3.resolution
BATCH = 128
#: Processes that read and cut files (cutting holds Python's lock: threads ran on one core).
READERS = 12
STATUS_EVERY_S = 30.0
INDEX = "pop-all"
EXACT_KEY = f"{INDEX}/a3_exact.npy"
EXACT_SCALE_KEY = f"{INDEX}/a3_exact_scale.npy"
DIM = 8448


def file_pixels(s3, bucket: str, file_id: str, grid: Grid) -> np.ndarray | str:
    """Every window of the file, as uint8 (windows, 322, 322, 3), or why it cannot be read."""
    try:
        data = s3.get_object(Bucket=bucket, Key=full_key(file_id))["Body"].read()
    except Exception as err:
        return f"read: {type(err).__name__}"
    try:
        with PILImage.open(io.BytesIO(data)) as opened:
            image = opened.convert("RGB")
        windows = [
            as_gallery_crop(image.crop(box)).resize((SIDE, SIDE), PILImage.Resampling.BILINEAR)
            for box in crop_boxes(image.width, image.height, grid)
        ]
        return np.stack([np.asarray(w, dtype=np.uint8) for w in windows])
    except Exception as err:
        return f"unreadable: {type(err).__name__}"


_CLIENT: list[object] = []


def load_file(job: tuple[str, str, str, Grid]) -> tuple[str, str, np.ndarray | str | None]:
    """`file_pixels` in a worker process, with that process's own Object Storage client;
    an empty file id (a piece's end) passes through."""
    piece, file_id, bucket, grid = job
    if not file_id:
        return piece, file_id, None
    if not _CLIENT:
        _CLIENT.append(s3_client(get_settings()))
    return piece, file_id, file_pixels(_CLIENT[0], bucket, file_id, grid)


class Megaloc:
    """MegaLoc on the GPU, in any `Precision`; pixels normalised there as `squash` does."""

    def __init__(self) -> None:
        self.model = torch.hub.load(A3.model, "get_trained_model", trust_repo=True)
        self.model = self.model.to(DEVICE).eval()
        self.mean = torch.from_numpy(IMAGENET_MEAN).to(DEVICE).view(1, 3, 1, 1)
        self.std = torch.from_numpy(IMAGENET_STD).to(DEVICE).view(1, 3, 1, 1)
        self.bf16 = torch.cuda.get_device_capability()[0] >= 8
        self.seconds = 0.0

    def runs_as(self, precision: Precision) -> Precision:
        """bfloat16 needs an Ampere GPU or newer; an older one runs float16 instead."""
        return Precision.FP16 if precision is Precision.BF16 and not self.bf16 else precision

    @contextlib.contextmanager
    def _mode(self, precision: Precision) -> Iterator[None]:
        tf32 = precision is not Precision.FP32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        if precision in (Precision.BF16, Precision.FP16):
            dtype = torch.bfloat16 if precision is Precision.BF16 else torch.float16
            with torch.autocast("cuda", dtype=dtype):
                yield
        else:
            yield

    @torch.inference_mode()
    def encode(self, pixels: np.ndarray, precision: Precision) -> np.ndarray:
        """Unit vectors (float32) of uint8 windows (n, 322, 322, 3)."""
        start = time.monotonic()
        out = []
        for at in range(0, len(pixels), BATCH):
            block = torch.from_numpy(pixels[at : at + BATCH]).to(DEVICE)
            x = (block.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
            with self._mode(self.runs_as(precision)):
                v = self.model(x.contiguous())
            v = v.float()
            out.append(torch.nn.functional.normalize(v, dim=1).cpu().numpy())
        torch.cuda.synchronize()
        self.seconds += time.monotonic() - start
        vectors = np.concatenate(out) if out else np.zeros((0, DIM), np.float32)
        if not np.isfinite(vectors).all():
            msg = f"MegaLoc returned NaN or infinite values in {precision}"
            raise FloatingPointError(msg)
        return vectors


class Projector:
    """`ProjectedCrops` on the GPU: C (x - m) and m . x of unit rows."""

    def __init__(self, s3, bucket: str, spec: CropSpec) -> None:
        body = s3.get_object(Bucket=bucket, Key=spec.projection_key)["Body"].read()
        with np.load(io.BytesIO(body), allow_pickle=False) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
            components = np.asarray(data["components"], dtype=np.float32)[: spec.dims]
        self.mean = torch.from_numpy(mean).to(DEVICE)
        self.components = torch.from_numpy(components).to(DEVICE)

    @torch.inference_mode()
    def apply(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(vectors).to(DEVICE)
        coords = (x - self.mean) @ self.components.T
        return coords.half().cpu().numpy(), (x @ self.mean).cpu().numpy()


class Piece:
    """One piece's files and vectors, gathered while its files come in."""

    def __init__(self, name: str, spec: CropSpec) -> None:
        self.name = name
        self.spec = spec
        self.files: list[str] = []
        self.failed: list[str] = []
        self.reasons: list[str] = []
        self.parts: dict[str, list[np.ndarray]] = {}
        self.whole: list[np.ndarray] = []
        self.pending: list[np.ndarray] = []

    def add(self, file_id: str, pixels: np.ndarray | str) -> None:
        if isinstance(pixels, str):
            self.failed.append(file_id)
            self.reasons.append(pixels)
        else:
            self.files.append(file_id)
            self.pending.append(pixels)

    def waiting(self) -> int:
        return sum(len(p) for p in self.pending)

    def embed(self, model: Megaloc, projector: Projector | None) -> int:
        """Embed the windows waiting; returns how many."""
        if not self.pending:
            return 0
        pixels = np.concatenate(self.pending)
        self.pending = []
        for precision in self.spec.precisions:
            vectors = model.encode(pixels, precision)
            if self.spec.form is Form.WHOLE:
                key = f"vectors_{model.runs_as(precision)}"
                self.parts.setdefault(key, []).append(vectors.astype(np.float16))
            else:
                assert projector is not None
                coords, mean_dot = projector.apply(vectors)
                self.parts.setdefault("coords", []).append(coords)
                self.parts.setdefault("mean_dot", []).append(mean_dot)
                if self.spec.keep_whole:
                    rows, scale = quantize_rows(vectors)
                    self.whole.append(rows)
                    self.parts.setdefault("whole_scale", []).append(scale)
                break  # the full run keeps its first precision only
        return len(pixels)

    def body(self, per_file: int) -> bytes:
        arrays = {k: np.concatenate(v) for k, v in self.parts.items()}
        for key, value in arrays.items():
            if len(value) != len(self.files) * per_file:
                msg = f"piece {self.name}: {key} has {len(value)} rows for {len(self.files)} files"
                raise RuntimeError(msg)
        arrays["failed"] = np.array(self.failed, dtype=str)
        arrays["reasons"] = np.array(self.reasons, dtype=str)
        return write_piece(self.files, per_file, arrays)

    def whole_body(self) -> bytes:
        rows = np.concatenate(self.whole) if self.whole else np.zeros((0, DIM), np.int8)
        buffer = io.BytesIO()
        np.save(buffer, rows, allow_pickle=False)
        return buffer.getvalue()


class Reporter:
    """The run's status file (`JobStatus`), written at most every `STATUS_EVERY_S`."""

    def __init__(self, s3, bucket: str, status: JobStatus) -> None:
        self.s3 = s3
        self.bucket = bucket
        self.status = status
        self.counts: Counter[str] = Counter()
        self.reported = 0.0

    def set(self, *, force: bool = False, **changes: object) -> None:
        self.status = self.status.now(counts=dict(self.counts), **changes)
        if not force and time.time() - self.reported < STATUS_EVERY_S:
            return
        self.reported = time.time()
        key = status_key(self.status.job, 0)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=self.status.to_json().encode())
        logger.info("%s", self.status.line())


def run_crop_embed(settings: Settings, *, name: str, first: int, last: int) -> JobStatus:
    """Pieces `first`..`last` of the job `name` that are not written yet."""
    s3 = s3_client(settings)
    bucket = bucket_name(settings, Bucket.IMAGES)
    spec = read_crop_spec(s3.get_object(Bucket=bucket, Key=spec_key(name))["Body"].read())
    pieces = spec.pieces()[first : last + 1]
    done = written_pieces(s3, bucket, name)
    todo = [p for p in pieces if p not in done]
    note = f"{len(pieces) - len(todo)} pieces already written"
    reporter = Reporter(
        s3, bucket, JobStatus(job_name(name, first, last), 0, 1, total=len(todo), note=note)
    )
    try:
        reporter.set(force=True, note="weights")
        download_tree(s3, bucket_name(settings, Bucket.WEIGHTS), Path(WEIGHTS_MOUNT))
        model = Megaloc()
        logger.info("GPU %s; %d pieces to do", torch.cuda.get_device_name(), len(todo))
        if spec.queries and first == 0:
            reporter.set(force=True, note="queries")
            _queries(s3, bucket, bucket_name(settings, Bucket.INDEX), spec, model)
        if spec.trial and todo:
            reporter.set(force=True, note="trial")
            _trial(s3, bucket, spec, model, todo[0])
        projector = Projector(s3, bucket, spec) if spec.form is Form.PROJECTED else None
        _run(s3, bucket, spec=spec, model=model, projector=projector, todo=todo, reporter=reporter)
    except Exception as err:
        reporter.set(force=True, state=JobState.FAILED, note=f"{type(err).__name__}: {err}"[:500])
        raise
    reporter.set(force=True, state=JobState.DONE)
    return reporter.status


def _files_of(s3, bucket: str, spec: CropSpec, piece: str) -> list[str]:
    if spec.files:
        return spec.piece_files(piece)
    keys = listing(s3, bucket, f"{FULL_PREFIX}/{piece[:2]}/{piece}")
    return sorted(Path(k.key).stem for k in keys)


def _run(
    s3,
    bucket: str,
    *,
    spec: CropSpec,
    model: Megaloc,
    projector: Projector | None,
    todo: list[str],
    reporter: Reporter,
) -> None:
    """Every file of every piece in `todo`, read ahead of the GPU on `READERS` threads; a
    piece is written when its last file is embedded."""
    per_file = len(crop_boxes(100, 100, spec.grid))

    def work() -> Iterator[tuple[str, str]]:
        for piece in todo:
            yield from ((piece, f) for f in _files_of(s3, bucket, spec, piece))
            yield piece, ""  # the piece ends here

    started = time.monotonic()
    crops = written = 0
    current: Piece | None = None
    jobs = ((piece, file_id, bucket, spec.grid) for piece, file_id in work())
    loaded = process_map(load_file, jobs, workers=READERS, window=4 * READERS)
    for piece, file_id, pixels in loaded:
        if current is None:
            current = Piece(piece, spec)
        if pixels is None:
            crops += current.embed(model, projector)
            body = current.body(per_file)
            if spec.keep_whole:  # first: a piece counts as done once its coordinates are in
                whole = current.whole_body()
                s3.put_object(Bucket=bucket, Key=whole_key(spec.name, piece), Body=whole)
            s3.put_object(Bucket=bucket, Key=piece_key(spec.name, piece), Body=body)
            written += 1
            current = None
        else:
            current.add(file_id, pixels)
            reporter.counts["ok" if not isinstance(pixels, str) else pixels.split(":")[0]] += 1
            if current.waiting() >= BATCH:
                crops += current.embed(model, projector)
        wall = time.monotonic() - started
        reporter.set(
            done=written,
            note=f"{crops:,} crops in {wall:.0f} s ({crops / max(wall, 1):.1f}/s), "
            f"GPU {model.seconds:.0f} s",
        )


def _trial(s3, bucket: str, spec: CropSpec, model: Megaloc, piece: str) -> None:
    """Time `spec.trial` crops in every precision; cosines against float32 (`trial_key`)."""
    files = _files_of(s3, bucket, spec, piece)
    stacks: list[np.ndarray] = []
    for pixels in bounded_map(
        lambda f: file_pixels(s3, bucket, f, spec.grid), files, workers=READERS
    ):
        if not isinstance(pixels, str):
            stacks.append(pixels)
        if sum(len(s) for s in stacks) >= spec.trial:
            break
    pixels = np.concatenate(stacks)[: spec.trial]
    model.encode(pixels[:BATCH], Precision.FP32)  # warm up
    out: dict[str, object] = {"crops": len(pixels), "gpu": torch.cuda.get_device_name()}
    reference: np.ndarray | None = None
    for precision in Precision:
        runs = model.runs_as(precision)
        model.encode(pixels[:BATCH], precision)  # warm up
        start = time.monotonic()
        vectors = model.encode(pixels, precision)
        per_s = len(pixels) / (time.monotonic() - start)
        if reference is None:
            reference = vectors
        cosines = (vectors * reference).sum(axis=1)
        out[str(precision)] = {
            "runs_as": str(runs),
            "crops_per_s": round(per_s, 1),
            "cosine_min": float(cosines.min()),
            "cosine_p1": float(np.percentile(cosines, 1)),
            "cosine_median": float(np.median(cosines)),
        }
        logger.info(
            "trial %s (as %s): %.1f crops/s, cosine min %.5f", precision, runs, per_s, cosines.min()
        )
    model.seconds = 0.0
    s3.put_object(Bucket=bucket, Key=trial_key(spec.name), Body=json.dumps(out, indent=1).encode())


def _exact_rows(s3, index_bucket: str, rows: list[int]) -> np.ndarray:
    """Rows of the `pop-all` int8 store, as the site reads them (row * scale), by byte range:
    the 19 GB file is never downloaded."""
    head = s3.get_object(Bucket=index_bucket, Key=EXACT_KEY, Range="bytes=0-4095")["Body"].read()
    stream = io.BytesIO(head)
    version = np.lib.format.read_magic(stream)
    read_header = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    shape, _, dtype = read_header(stream)
    if dtype != np.int8 or shape[1] != DIM:
        msg = f"{EXACT_KEY}: int8 rows of {DIM} expected, got {dtype} {shape}"
        raise ValueError(msg)
    offset = stream.tell()
    body = s3.get_object(Bucket=index_bucket, Key=EXACT_SCALE_KEY)["Body"].read()
    scale = np.load(io.BytesIO(body), allow_pickle=False)

    def row(r: int) -> np.ndarray:
        start = offset + r * DIM
        span = f"bytes={start}-{start + DIM - 1}"
        data = s3.get_object(Bucket=index_bucket, Key=EXACT_KEY, Range=span)["Body"].read()
        return np.frombuffer(data, dtype=np.int8).astype(np.float32) * scale[r]

    return np.stack(list(bounded_map(row, rows, workers=READERS)))


def _queries(s3, bucket: str, index_bucket: str, spec: CropSpec, model: Megaloc) -> None:
    """Every slice query's unit MegaLoc vector (`queries_key`)."""
    with np.load(io.BytesIO(s3.get_object(Bucket=bucket, Key=SUPPLIED_KEY)["Body"].read())) as data:
        supplied = {
            str(i): np.asarray(data["a3"][k], np.float32)
            for k, i in enumerate(data["ids"].tolist())
        }
    catalog = [q for q in spec.queries if q.source is QuerySource.CATALOG]
    stored = dict(
        zip(
            [q.query_id for q in catalog],
            _exact_rows(s3, index_bucket, [q.row for q in catalog]),
            strict=True,
        )
    )
    vectors: list[np.ndarray] = []
    for query in spec.queries:
        if query.source is QuerySource.CATALOG:
            v = stored[query.query_id]
        elif query.source is QuerySource.SUPPLIED:
            v = supplied[query.image_id]
        else:
            jpeg = prepare_photo(
                s3.get_object(Bucket=bucket, Key=outside_key(query.image_id))["Body"].read()
            )
            with PILImage.open(io.BytesIO(jpeg)) as opened:
                pixels = squash(opened.convert("RGB"), SIDE)
            with torch.inference_mode():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                v = model.model(torch.from_numpy(pixels[None]).to(DEVICE)).float().cpu().numpy()[0]
        vectors.append(np.asarray(v, np.float32) / np.linalg.norm(v))
    buffer = io.BytesIO()
    np.savez(buffer, query_ids=np.array([q.query_id for q in spec.queries]), a3=np.stack(vectors))
    s3.put_object(Bucket=bucket, Key=queries_key(spec.name), Body=buffer.getvalue())
    logger.info("queries: %d vectors (%d stored rows read)", len(vectors), len(catalog))
