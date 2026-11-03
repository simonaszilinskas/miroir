"""Crop vectors on Scaleway GPU machines, kept (docs/search-levels.md, step 4).

The crop slice test (`miroir.jobs.crop_slice`) scored crops and threw their vectors away. This
job keeps them, so the Mac can try shorter forms and the index can serve them:

- **The slice** (`slice_spec`): the 3x3 crops of the slice's gallery, every vector whole
  (8,448 numbers, float16) in each precision of `CropSpec.precisions`, and the slice queries'
  MegaLoc vectors. The Mac shortens them every way it wants to compare
  (`miroir.eval.crop_short`).
- **All of POP** (`full_spec`): the 3x3 crops of every stored file, cut into the 4,096 pieces
  of the file ids (`miroir.jobs.bulk.piece_prefix`), stored projected on the first `dims`
  principal axes of a PCA (`CropSpec.projection_key`), with what the scorer needs to put a
  crop's cosine back on the scale of a whole cosine (`ProjectedCrops`).

The Mac writes a `CropSpec` to the images bucket (`spec_key`); each machine
(`miroir.remote.crop_embed`) does a range of pieces and writes one file per piece
(`piece_key`), so a rerun, on any machine, skips the pieces already written. Nothing here
loads a model.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter

from miroir.config import Settings
from miroir.jobs.bulk import PIECES, piece_prefix
from miroir.jobs.crop_slice import Grid, QuerySource, SliceSpec, read_spec
from miroir.jobs.objects import Bucket, bucket_name, listing

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

    from miroir.jobs.scaleway import InstanceApi, Machine, Offer
    from miroir.jobs.scw_bulk import Launch
    from miroir.jobs.scw_run import Shell

logger = logging.getLogger(__name__)

PREFIX = "crops"
#: The grid of the full run: 3x3 windows of 40% of each side (crop slice test, variant d).
GRID = Grid(3, 0.4)
#: Files per piece of the slice (the full run uses the 4,096 file-id pieces).
SLICE_PIECE_FILES = 500
#: Names of the two jobs.
SLICE_NAME = "slice-3x3"
FULL_NAME = "pop-3x3"


class Precision(StrEnum):
    """How MegaLoc runs on the GPU. The gallery and the site's queries ran in float32."""

    FP32 = "fp32"
    TF32 = "tf32"
    """float32 numbers, matrix products on the tensor cores with 10-bit mantissas."""
    BF16 = "bf16"
    FP16 = "fp16"


class Form(StrEnum):
    WHOLE = "whole"
    """Every 8,448-d unit vector, float16."""
    PROJECTED = "projected"
    """`ProjectedCrops`: the first `dims` principal coordinates and the dot with the mean."""


@dataclass(frozen=True, slots=True)
class SliceQueryInput:
    """A slice query whose MegaLoc vector the machine fetches or makes."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    query_id: str
    source: QuerySource
    image_id: str
    row: int
    """Its row in the `pop-all` index (catalog queries); -1 otherwise."""


@dataclass(frozen=True, slots=True)
class CropSpec:
    __pydantic_config__ = ConfigDict(extra="forbid")

    name: str
    grid: Grid
    precisions: tuple[Precision, ...]
    """Each crop is embedded once per precision; the first is the one kept by the full run."""
    form: Form
    files: tuple[str, ...] = ()
    """The files to crop, cut into pieces of `SLICE_PIECE_FILES`; empty: every stored file,
    by file-id piece."""
    projection_key: str = ""
    """Images-bucket key of an `.npz` with `mean` (8,448) and `components` (dims x 8,448,
    largest axis first), for `Form.PROJECTED`."""
    dims: int = 0
    keep_whole: bool = False
    """`Form.PROJECTED` only: also keep every vector whole in int8 (`whole_key`), for the
    site to rescore the crops the short coordinates shortlist."""
    queries: tuple[SliceQueryInput, ...] = ()
    trial: int = 0
    """Crops timed in every precision before the run, against float32 (`trial.json`)."""
    note: str = ""

    def pieces(self) -> list[str]:
        """Every piece's name, in order."""
        if self.files:
            count = -(-len(self.files) // SLICE_PIECE_FILES)
            return [f"{p:04d}" for p in range(count)]
        return [piece_prefix(p) for p in range(PIECES)]

    def piece_files(self, piece: str) -> list[str]:
        """The files of a listed piece (a file-id piece is listed on the machine)."""
        at = int(piece) * SLICE_PIECE_FILES
        return list(self.files[at : at + SLICE_PIECE_FILES])


_SPEC = TypeAdapter(CropSpec)


def spec_json(spec: CropSpec) -> bytes:
    return _SPEC.dump_json(spec)


def read_crop_spec(data: bytes) -> CropSpec:
    return _SPEC.validate_json(data)


def spec_key(name: str) -> str:
    return f"{PREFIX}/{name}/spec.json"


def piece_key(name: str, piece: str) -> str:
    return f"{PREFIX}/{name}/pieces/{piece}.npz"


def whole_key(name: str, piece: str) -> str:
    """A piece's whole vectors: int8 rows in the piece's order, a plain `.npy` that the site
    memory-maps; their scales are the piece's `whole_scale`."""
    return f"{PREFIX}/{name}/whole/{piece}.npy"


def queries_key(name: str) -> str:
    return f"{PREFIX}/{name}/queries.npz"


def trial_key(name: str) -> str:
    return f"{PREFIX}/{name}/trial.json"


def job_name(name: str, first: int, last: int, *, reverse: bool = False) -> str:
    """The status file's job: one per machine's range of pieces and direction."""
    return f"crops-{name}-{first}-{last}" + ("-reverse" if reverse else "")


# --- what a piece holds ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProjectedCrops:
    """Crops kept short. For a unit crop vector x, a unit query q, the PCA's mean m and its
    first axes C: `coords` = C (x - m), `mean_dot` = m . x. Then

        x . q  ~  coords . C (q - m) + m . x + m . q - m . m

    the whole cosine, short of the part of (x - m) . (q - m) outside the kept axes: a crop
    scores on the scale of a whole image's exact cosine (`approx_cosines`). Keeping fewer
    axes is taking fewer columns of `coords`."""

    files: tuple[str, ...]
    per_file: int
    """Crops per file, in grid order; crop k of file f is row f * per_file + k."""
    coords: NDArray[np.float16]
    mean_dot: NDArray[np.float32]


def project_crops(
    vectors: NDArray[np.floating], mean: NDArray[np.float32], components: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """(coords, mean_dot) of unit rows (`ProjectedCrops`)."""
    x = np.asarray(vectors, dtype=np.float32)
    return ((x - mean) @ components.T).astype(np.float32), (x @ mean).astype(np.float32)


def query_terms(
    query: NDArray[np.floating], mean: NDArray[np.float32], components: NDArray[np.float32]
) -> tuple[NDArray[np.float32], float]:
    """A unit query's side of `approx_cosines`: C (q - m), and m . q - m . m."""
    q = np.asarray(query, dtype=np.float32)
    return ((q - mean) @ components.T).astype(np.float32), float(q @ mean - mean @ mean)


def approx_cosines(
    coords: NDArray[np.floating],
    mean_dot: NDArray[np.floating],
    query_coords: NDArray[np.floating],
    query_offset: float,
) -> NDArray[np.float32]:
    """Each crop's approximate whole cosine with one query (`ProjectedCrops`); `coords` and
    `query_coords` cut to the same number of axes."""
    dims = query_coords.shape[-1]
    return (
        np.asarray(coords[:, :dims], dtype=np.float32) @ np.asarray(query_coords, np.float32)
        + np.asarray(mean_dot, dtype=np.float32)
        + query_offset
    ).astype(np.float32)


def write_piece(
    files: Sequence[str], per_file: int, arrays: dict[str, NDArray[np.generic]]
) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, files=np.array(list(files), dtype=str), per_file=np.int64(per_file), **arrays)  # pyright: ignore[reportArgumentType]
    return buffer.getvalue()


# --- specs ---------------------------------------------------------------------------------


def slice_spec(slice_spec_path: Path, index_ids: Sequence[str]) -> CropSpec:
    """3x3 crops of the crop slice's stored files, whole, in float32 and bfloat16, with the
    slice's queries (`miroir.jobs.crop_slice`)."""
    old: SliceSpec = read_spec(slice_spec_path.read_bytes())
    row = {image_id: i for i, image_id in enumerate(index_ids)}
    queries = tuple(
        SliceQueryInput(
            query_id=q.query_id,
            source=q.source,
            image_id=q.image_id,
            row=row[q.image_id] if q.source is QuerySource.CATALOG else -1,
        )
        for q in old.queries
    )
    files = tuple(sorted({f for f in old.files if f}))
    return CropSpec(
        name=SLICE_NAME,
        grid=GRID,
        precisions=(Precision.FP32, Precision.BF16),
        form=Form.WHOLE,
        files=files,
        queries=queries,
        trial=1024,
        note=f"3x3 crops of the crop slice's {len(files):,} stored files",
    )


def full_spec(precision: Precision, projection_key: str, dims: int) -> CropSpec:
    """Every stored file: PCA coordinates to shortlist, whole int8 vectors to rescore (the
    short ones alone lose T2 and T3 queries to look-alikes: docs/results.md, "Short crops")."""
    return CropSpec(
        name=FULL_NAME,
        grid=GRID,
        precisions=(precision,),
        form=Form.PROJECTED,
        projection_key=projection_key,
        dims=dims,
        keep_whole=True,
        note=f"3x3 crops of every stored POP file, {precision} MegaLoc, PCA {dims} and whole int8",
    )


def put_spec(settings: Settings, s3: S3Client, spec: CropSpec) -> None:
    bucket = bucket_name(settings, Bucket.IMAGES)
    s3.put_object(Bucket=bucket, Key=spec_key(spec.name), Body=spec_json(spec))


def written_pieces(s3: S3Client, bucket: str, name: str) -> set[str]:
    return {Path(s.key).stem for s in listing(s3, bucket, f"{PREFIX}/{name}/pieces/")}


def crops_dir(settings: Settings, name: str) -> Path:
    return settings.scaleway_dir / PREFIX / name


def fetch_job(settings: Settings, s3: S3Client, name: str) -> list[Path]:
    """Copy the job's pieces, queries, trial and projection to `crops_dir` (those already
    there are kept). The whole vectors of the full run (`whole_key`) stay in the bucket."""
    from miroir.jobs.objects import download_keys, download_tree  # noqa: PLC0415

    folder = crops_dir(settings, name)
    folder.mkdir(parents=True, exist_ok=True)
    bucket = bucket_name(settings, Bucket.IMAGES)
    root = folder.parent.parent
    download_tree(s3, bucket, root, prefix=f"{PREFIX}/{name}/pieces/")
    singles = (queries_key(name), trial_key(name), f"{PREFIX}/{name}/projection.npz")
    download_keys(s3, bucket, root, singles)
    return sorted(folder.rglob("*.npz"))


# --- machines --------------------------------------------------------------------------------


#: Offers a crop run may use (prices as `miroir.jobs.scaleway.OFFERS`), best value first by
#: the speeds measured on the slice: MegaLoc's GPU time, not the machine's other parts,
#: sets the cost of a crop.
def crop_offers() -> list[Offer]:
    from miroir.jobs.crop_slice import RENDER_OFFER  # noqa: PLC0415
    from miroir.jobs.scaleway import Offer  # noqa: PLC0415

    return [
        Offer("fr-par-2", "L40S-1-48G", 1.47),
        Offer("pl-waw-2", "L40S-1-48G", 1.47),
        Offer("fr-par-2", "L4-1-24G", 0.79),
        Offer("pl-waw-2", "L4-1-24G", 0.79),
        Offer("fr-par-1", "L4-1-24G", 0.79),
        RENDER_OFFER,
    ]


def in_stock(api: InstanceApi, offers: Sequence[Offer]) -> list[Offer]:
    """The offers whose type Scaleway does not report short ("shortage") in their zone, in
    their own order: a "scarce" fast GPU is worth a try before an "available" slow one, and
    one out of stock fails at once (`gpu_machine` then tries the next)."""
    states: dict[str, dict[str, str]] = {}
    found: list[Offer] = []
    for offer in offers:
        if offer.zone not in states:
            states[offer.zone] = api.availability(offer.zone)
        state = states[offer.zone].get(offer.commercial_type, "shortage")
        logger.info("%s %s: %s", offer.zone, offer.commercial_type, state)
        if state in {"available", "scarce"}:
            found.append(offer)
    return found


def crop_launch(
    settings: Settings, name: str, first: int, last: int, torch_tag: str = ""
) -> Launch:
    """`miroir scw crop-embed-run` for pieces `first`..`last` in the job image."""
    from miroir.jobs.scw_bulk import Launch  # noqa: PLC0415
    from miroir.jobs.scw_run import IMAGE_TAG, MOUNTS, REMOTE_ROOT, build_image  # noqa: PLC0415

    def start(shell: Shell, machine: Machine) -> None:
        tag = "cu126" if machine.offer.commercial_type == "RENDER-S" else ""
        build_image(shell, settings, torch_tag=tag)
        run = f"docker run -d --env-file {REMOTE_ROOT}/env {MOUNTS} --entrypoint miroir "
        shell.run(
            f"{run} --name watch {IMAGE_TAG} scw watch --server-id {machine.server_id} "
            f"--zone {machine.offer.zone} --deadline {machine.deadline:.0f}"
        )
        shell.run(
            f"{run} --name crops --gpus all --shm-size 8g {IMAGE_TAG} scw crop-embed-run "
            f"--name {name} --first {first} --last {last}" + (" --reverse" if reverse else "")
        )

    return Launch(
        job=job_name(name, first, last, reverse=reverse),
        shards=(0,),
        start=start,
        alive="docker ps -q --filter name=^crops$ | wc -l",
        logs="docker logs --tail 40 crops 2>&1",
        probe="nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader "
        "&& cat /proc/loadavg",
    )
