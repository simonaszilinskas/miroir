"""The crop slice test (docs/search-levels.md, "Crop slice test"): do crop vectors help find a
whole from a photo of one of its parts, before paying for crops of every POP image?

The Mac picks a gallery of about 50,000 images of the `pop-all` index (every image of the
test sets' answers, plus a seeded sample of others) and writes the spec to the images bucket
(`SPEC_KEY`), with the vectors of queries that are in no POP record (`SUPPLIED_KEY`) and the
photos of those only a model can read (`outside_key`). A GPU machine
(`miroir.remote.crop_slice`) cuts each gallery photo into the `GRIDS` crops, embeds them,
scores them against every query, also scores the queries against the whole `pop-all` index,
and writes score tables back (`out_key`). The Mac ranks with them (`miroir.eval.crops`).
Nothing here loads a model; the machine and the Mac both import it.
"""

from __future__ import annotations

import io
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
from pydantic import ConfigDict, TypeAdapter

from miroir.config import SEED, Settings
from miroir.jobs.objects import Bucket, bucket_name
from miroir.jobs.scaleway import Offer

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

    from miroir.eval.sets import Query
    from miroir.jobs.scaleway import Machine
    from miroir.jobs.scw_bulk import Launch
    from miroir.jobs.scw_run import Shell

logger = logging.getLogger(__name__)

JOB = "crop-slice"
INDEX = "pop-all"
#: The image rows with their files (`miroir.jobs.scw_bulk.IMAGE_FILES`).
IMAGE_FILES = "image-files.parquet"
PREFIX = "crop-slice"
SPEC_KEY = f"{PREFIX}/in/spec.json"
SUPPLIED_KEY = f"{PREFIX}/in/supplied.npz"
#: Gallery size the test aims at, and the sample that also gets EmbeddingGemma 2 crops.
GALLERY_SIZE = 50_000
A7_DISTRACTORS = 3_000
#: Sets whose answers get EmbeddingGemma 2 crops: their parts are often black-and-white
#: Mémoire photos, which the site ranks by EmbeddingGemma 2 alone (archive-building bucket).
A7_SETS = ("VET", "PARTS")


def outside_key(image_id: str) -> str:
    return f"{PREFIX}/in/outside/{image_id}.jpg"


def out_key(name: str) -> str:
    return f"{PREFIX}/out/{name}.npz"


@dataclass(frozen=True, slots=True)
class Grid:
    """`n` x `n` windows, each `side` of the photo's width and height, spread evenly from edge
    to edge. A side above 1/n makes neighbours overlap, so a part lying on a line between two
    cells is still whole in one window."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    n: int
    side: float

    def __post_init__(self) -> None:
        if self.n < 2 or not 1 / self.n <= self.side < 1:
            msg = f"a {self.n}x{self.n} grid needs a side in [1/{self.n}, 1), got {self.side}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return f"g{self.n}"


#: 2x2 windows of 60% (20% shared between neighbours), 3x3 windows of 40% (10% shared).
GRIDS = (Grid(2, 0.6), Grid(3, 0.4))


def crop_boxes(width: int, height: int, grid: Grid) -> list[tuple[int, int, int, int]]:
    """(left, top, right, bottom) of each window, row by row, in pixels."""
    step = (1 - grid.side) / (grid.n - 1)
    boxes: list[tuple[int, int, int, int]] = []
    for row in range(grid.n):
        for col in range(grid.n):
            left, top = col * step, row * step
            boxes.append(
                (
                    round(left * width),
                    round(top * height),
                    round((left + grid.side) * width),
                    round((top + grid.side) * height),
                )
            )
    return boxes


class QuerySource(StrEnum):
    CATALOG = "catalog"
    """A POP image of the index: its stored vectors, hidden from its own search."""
    SUPPLIED = "supplied"
    """A photo in no record whose vectors are on the Mac (Commons photos of T2)."""
    OUTSIDE = "outside"
    """A photo in no record with no vectors yet (VET): the machine embeds it."""


@dataclass(frozen=True, slots=True)
class SliceQuery:
    __pydantic_config__ = ConfigDict(extra="forbid")

    query_id: str
    set_id: str
    source: QuerySource
    image_id: str
    file_id: str
    """The stored file of a catalog query (for its chroma spread); empty otherwise."""
    hidden: tuple[str, ...]
    """Image rows this query may not find (itself, other details of its entity)."""
    expected: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SliceSpec:
    __pydantic_config__ = ConfigDict(extra="forbid")

    gallery: tuple[str, ...]
    """Image rows of the index (`ids.txt`) in the slice."""
    files: tuple[str, ...]
    """The stored file of each gallery row; empty for a Commons photo (no stored file, so no
    crops: it keeps its whole vectors only)."""
    a7_files: tuple[str, ...]
    """Files that also get EmbeddingGemma 2 crops."""
    queries: tuple[SliceQuery, ...]
    grids: tuple[Grid, ...] = GRIDS
    seed: int = SEED
    note: str = ""

    def crop_files(self) -> list[str]:
        """Each stored file once: the EmbeddingGemma 2 sample first, then the others, each
        part in a seeded order, so a run cut short leaves a random part of the gallery
        without crops, not the sample of others alone."""
        rng = random.Random(self.seed)
        first = sorted(dict.fromkeys(self.a7_files))
        seen = set(first)
        rest = sorted({f for f in self.files if f and f not in seen})
        rng.shuffle(first)
        rng.shuffle(rest)
        return first + rest


_SPEC = TypeAdapter(SliceSpec)


def spec_json(spec: SliceSpec) -> bytes:
    return _SPEC.dump_json(spec)


def read_spec(data: bytes) -> SliceSpec:
    return _SPEC.validate_json(data)


def pick_gallery(
    index_ids: Sequence[str],
    entity_of: Mapping[str, str],
    targets: set[str],
    queries: set[str],
    *,
    size: int = GALLERY_SIZE,
    seed: int = SEED,
) -> tuple[list[str], list[str]]:
    """(answer rows, sampled other rows): every index row of a `targets` entity, then a seeded
    sample of the other rows up to `size` in all. Query photos never enter the gallery."""
    answers = [i for i in index_ids if entity_of.get(i) in targets and i not in queries]
    others = [i for i in index_ids if entity_of.get(i) not in targets and i not in queries]
    wanted = max(0, min(size - len(answers), len(others)))
    sample = random.Random(seed).sample(others, wanted)
    return answers, sorted(sample)


def pick_a7_files(
    answer_files: Sequence[str], other_files: Sequence[str], count: int, seed: int = SEED
) -> list[str]:
    """The answers' files, then `count` seeded files of the others (stored files only)."""
    answers = [f for f in dict.fromkeys(answer_files) if f]
    taken = set(answers)
    pool = sorted({f for f in other_files if f and f not in taken})
    return answers + sorted(random.Random(seed).sample(pool, min(count, len(pool))))


# --- on the Mac -------------------------------------------------------------------------------

#: Test sets of the slice, tune halves only (VET has no other).
SETS = ("VET", "PARTS", "T2", "T3")
#: Commons query photos' vectors (T2), embedded with the July runs.
QUERY_RUNS = (
    "A7-embeddinggemma-2-512-T2-T2b-T8c-tune-queries",
    "A3-megaloc-322-T2-T2b-T8c-tune-queries",
)
OUT_NAMES = ("queries", "full", "whole", "crops")


def slice_dir(settings: Settings) -> Path:
    """Inputs and returned scores of the test, beside the other Scaleway files."""
    return settings.scaleway_dir / PREFIX


def source_of(image_id: str, index: set[str]) -> QuerySource:
    from miroir.eval.parts import OUTSIDE_PREFIX  # noqa: PLC0415 (the Mac's side only)

    if image_id in index:
        return QuerySource.CATALOG
    if image_id.startswith(OUTSIDE_PREFIX):
        return QuerySource.OUTSIDE
    return QuerySource.SUPPLIED


def build_spec(
    queries: Sequence[Query],
    index_ids: Sequence[str],
    entity_of: Mapping[str, str],
    file_of: Mapping[str, str],
    *,
    size: int = GALLERY_SIZE,
    a7_distractors: int = A7_DISTRACTORS,
    a7_sets: Sequence[str] = A7_SETS,
    seed: int = SEED,
) -> SliceSpec:
    """The gallery, its files and the queries (`pick_gallery`, `pick_a7_files`)."""
    from miroir.commons.model import is_commons_image  # noqa: PLC0415 (the Mac's side only)

    index = set(index_ids)
    targets = {e for q in queries for e in q.expected_entity_ids}
    asking = {q.query_image_id for q in queries}
    answers, others = pick_gallery(index_ids, entity_of, targets, asking, size=size, seed=seed)
    gallery = answers + others

    def file(image_id: str) -> str:
        return "" if is_commons_image(image_id) else file_of.get(image_id, image_id)

    a7_targets = {e for q in queries if q.set_id in a7_sets for e in q.expected_entity_ids}
    a7_files = (
        []
        if not a7_sets
        else pick_a7_files(
            [file(i) for i in answers if entity_of.get(i) in a7_targets],
            [file(i) for i in others],
            a7_distractors,
            seed,
        )
    )
    picked = [
        SliceQuery(
            query_id=q.query_id,
            set_id=q.set_id,
            source=(source := source_of(q.query_image_id, index)),
            image_id=q.query_image_id,
            file_id=file(q.query_image_id) if source is QuerySource.CATALOG else "",
            hidden=tuple(dict.fromkeys((q.query_image_id, *q.exclude_image_ids))),
            expected=q.expected_entity_ids,
        )
        for q in queries
    ]
    note = (
        f"{len(answers)} answer images, {len(others)} others sampled with seed {seed}; "
        f"{len(a7_files)} files with EmbeddingGemma 2 crops"
    )
    return SliceSpec(
        gallery=tuple(gallery),
        files=tuple(file(i) for i in gallery),
        a7_files=tuple(a7_files),
        queries=tuple(picked),
        seed=seed,
        note=note,
    )


def sample_set(queries: Sequence[Query], set_id: str, count: int, seed: int = SEED) -> list[Query]:
    """`count` seeded queries of `set_id`, every query of the other sets."""
    mine = [q for q in queries if q.set_id == set_id]
    kept = {q.query_id for q in random.Random(seed).sample(mine, min(count, len(mine)))}
    return [q for q in queries if q.set_id != set_id or q.query_id in kept]


def slice_queries(settings: Settings) -> list[Query]:
    from miroir.eval.sets import load_queries  # noqa: PLC0415 (the Mac's side only)

    return [q for s in SETS for q in load_queries(settings, s, "tune")]


def prepare(
    settings: Settings,
    s3: S3Client,
    *,
    a7_others: int = A7_DISTRACTORS,
    size: int = GALLERY_SIZE,
    t2_queries: int = 0,
) -> SliceSpec:
    """Pick the gallery from the `pop-all` index and send the spec, the supplied query
    vectors and the outside photos to the images bucket."""
    from miroir.eval.router3 import query_chroma  # noqa: PLC0415 (fetches photos)
    from miroir.eval.vectors import read_vectors  # noqa: PLC0415

    folder = slice_dir(settings)
    folder.mkdir(parents=True, exist_ok=True)
    ids_path = folder / "ids.txt"
    if not ids_path.exists():
        s3.download_file(bucket_name(settings, Bucket.INDEX), f"{INDEX}/ids.txt", str(ids_path))
    index_ids = ids_path.read_text(encoding="utf-8").split()
    tables = settings.scaleway_dir / "pop-all-tables"
    entity_of = _column_map(tables / "images.parquet", "image_id", "entity_id")
    entity_of.update(_column_map(settings.commons_gallery_parquet("full"), "image_id", "entity_id"))
    file_of = _column_map(settings.scaleway_dir / IMAGE_FILES, "image_id", "file_id")
    queries = slice_queries(settings)
    if t2_queries:
        queries = sample_set(queries, "T2", t2_queries)
    spec = build_spec(
        queries,
        index_ids,
        entity_of,
        file_of,
        size=size,
        a7_distractors=a7_others,
        a7_sets=A7_SETS if a7_others else (),
    )
    logger.info("%s", spec.note)

    bucket = bucket_name(settings, Bucket.IMAGES)
    supplied = [q.image_id for q in spec.queries if q.source is QuerySource.SUPPLIED]
    chroma = query_chroma(settings, supplied)
    runs = [read_vectors(settings.vectors_path(r)) for r in QUERY_RUNS]
    rows = [r.index() for r in runs]
    buffer = io.BytesIO()
    np.savez(
        buffer,
        ids=np.array(supplied),
        a7=np.stack([np.asarray(runs[0].vectors[rows[0][i]], np.float32) for i in supplied]),
        a3=np.stack([np.asarray(runs[1].vectors[rows[1][i]], np.float32) for i in supplied]),
        chroma=np.asarray([chroma[i] for i in supplied], dtype=np.float32),
    )
    s3.put_object(Bucket=bucket, Key=SUPPLIED_KEY, Body=buffer.getvalue())
    for q in spec.queries:
        if q.source is QuerySource.OUTSIDE:
            body = settings.outside_image_path(q.image_id).read_bytes()
            s3.put_object(Bucket=bucket, Key=outside_key(q.image_id), Body=body)
    s3.put_object(Bucket=bucket, Key=SPEC_KEY, Body=spec_json(spec))
    (folder / "spec.json").write_bytes(spec_json(spec))
    return spec


def vet_refs(settings: Settings, spec: SliceSpec) -> dict[str, str]:
    """The record of each gallery image of the VET answer (to single out its target photo)."""
    from miroir.eval.parts import VET_ENTITY  # noqa: PLC0415

    path = settings.scaleway_dir / "pop-all-tables" / "images.parquet"
    data = pq.read_table(path, columns=["image_id", "ref", "entity_id"]).to_pydict()  # pyright: ignore[reportUnknownMemberType]
    wanted = set(spec.gallery)
    return {
        i: r
        for i, r, e in zip(data["image_id"], data["ref"], data["entity_id"], strict=True)
        if e == VET_ENTITY and i in wanted
    }


def _column_map(path: Path, key: str, value: str) -> dict[str, str]:
    data = pq.read_table(path, columns=[key, value]).to_pydict()  # pyright: ignore[reportUnknownMemberType]
    return dict(zip(data[key], data[value], strict=True))


def fetch_outputs(settings: Settings, s3: S3Client) -> list[Path]:
    """Copy the machine's score tables to `slice_dir`; those it did not write are skipped."""
    from botocore.exceptions import ClientError  # noqa: PLC0415

    folder = slice_dir(settings)
    got: list[Path] = []
    for name in OUT_NAMES:
        path = folder / f"{name}.npz"
        try:
            s3.download_file(bucket_name(settings, Bucket.IMAGES), out_key(name), str(path))
        except ClientError:
            logger.warning("%s: not written by the machine", name)
            continue
        got.append(path)
    return got


# --- the machine -------------------------------------------------------------------------------


def crop_launch(settings: Settings, torch_tag: str = "") -> Launch:
    """`miroir scw crop-slice-run` in the job image on a GPU machine."""
    from miroir.jobs.scw_bulk import Launch  # noqa: PLC0415
    from miroir.jobs.scw_run import IMAGE_TAG, MOUNTS, REMOTE_ROOT, build_image  # noqa: PLC0415

    def start(shell: Shell, machine: Machine) -> None:
        build_image(shell, settings, torch_tag=torch_tag)
        run = f"docker run -d --env-file {REMOTE_ROOT}/env {MOUNTS} --entrypoint miroir "
        shell.run(
            f"{run} --name watch {IMAGE_TAG} scw watch --server-id {machine.server_id} "
            f"--zone {machine.offer.zone} --deadline {machine.deadline:.0f}"
        )
        shell.run(f"{run} --name crops --gpus all --shm-size 8g {IMAGE_TAG} scw crop-slice-run")

    return Launch(
        job=JOB,
        shards=(0,),
        start=start,
        alive="docker ps -q --filter name=^crops$ | wc -l",
        logs="docker logs --tail 40 crops 2>&1",
        probe="nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader "
        "&& cat /proc/loadavg && free -g | sed -n 2p",
    )


#: Scaleway's public price of a RENDER-S (Tesla P100 16 GB, 10 cores), rounded up: the only
#: GPU in stock on 2026-11-03, when every L4, L40S and H100 was short.
RENDER_OFFER = Offer("fr-par-2", "RENDER-S", 1.25)


def run_slice(
    settings: Settings,
    *,
    max_hours: float,
    l4: bool,
    render: bool,
    a7_others: int,
    size: int,
    t2_queries: int,
) -> list[Path]:
    """The crop slice test (docs/search-levels.md) on one machine: an L40S (an L4 with `l4`,
    a RENDER-S with MegaLoc crops only with `render`); the score tables it wrote come back to
    `slice_dir`, also when the run fails. Returns them."""
    from miroir.jobs.objects import s3_client  # noqa: PLC0415
    from miroir.jobs.scaleway import OFFERS  # noqa: PLC0415
    from miroir.jobs.scw_bulk import run_unattended  # noqa: PLC0415

    s3 = s3_client(settings)
    spec = prepare(
        settings, s3, a7_others=0 if render else a7_others, size=size, t2_queries=t2_queries
    )
    logger.info("%s", spec.note)
    offers = (
        [RENDER_OFFER] if render else [o for o in OFFERS if ("L40S" in o.commercial_type) != l4]
    )
    launch = crop_launch(settings, torch_tag="cu126" if render else "")
    try:
        run_unattended(settings, launch, offers=offers, max_hours=max_hours, purpose="crops")
    finally:
        got = fetch_outputs(settings, s3)
    return got
