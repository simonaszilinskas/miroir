"""Measure a bundle with crops against the same bundle without them, on a GPU machine.

The Mac writes the tune queries of `SETS` to the images bucket (`MeasureSpec`), with the
vectors of the Commons query photos it holds and the outside photos (VET). The machine
(`miroir.remote.crop_measure`) loads the bundle as the demo site does (`load_search`: gallery
and crops on the GPU, MegaLoc whole in int8 in memory), searches each query twice, crops off
then on, with the site's own code (`Engine.vector_cosines`, `photo_of`, `ranking.fuse`), and
writes back each query's rank of its right entity and the time each step took. A POP query
photo uses its stored vectors and is hidden from its own search, with the other images its
set hides. The Mac turns the ranks into hit rates (`summarize`). Nothing here loads a model.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import ConfigDict, TypeAdapter

from miroir.config import Settings
from miroir.jobs.crop_slice import IMAGE_FILES, QUERY_RUNS, QuerySource, SliceQuery, source_of
from miroir.jobs.objects import Bucket, bucket_name

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

    from miroir.eval.sets import Query
    from miroir.jobs.scaleway import Machine
    from miroir.jobs.scw_bulk import Launch
    from miroir.jobs.scw_run import Shell

logger = logging.getLogger(__name__)

JOB = "crop-measure"
PREFIX = "crop-measure"
SETS = ("T1", "T2", "T2b", "T3", "T5", "PARTS", "VET")
HIT_KS = (1, 10)


def spec_key(index: str) -> str:
    return f"{PREFIX}/{index}/spec.json"


def supplied_key(index: str) -> str:
    return f"{PREFIX}/{index}/supplied.npz"


def outside_key(index: str, image_id: str) -> str:
    return f"{PREFIX}/{index}/outside/{image_id}.jpg"


def out_key(index: str) -> str:
    return f"{PREFIX}/{index}/ranks.json"


@dataclass(frozen=True, slots=True)
class MeasureSpec:
    __pydantic_config__ = ConfigDict(extra="forbid")

    index: str
    queries: tuple[SliceQuery, ...]
    skipped: tuple[str, ...] = ()
    """Queries left out: a photo with neither stored nor supplied vectors."""


_SPEC = TypeAdapter(MeasureSpec)


def spec_json(spec: MeasureSpec) -> bytes:
    return _SPEC.dump_json(spec)


def read_measure_spec(data: bytes) -> MeasureSpec:
    return _SPEC.validate_json(data)


def measure_queries(
    queries: Sequence[Query],
    index_ids: set[str],
    file_of: Mapping[str, str],
    supplied: set[str],
    index: str,
) -> MeasureSpec:
    """Each query with where its vectors come from; those with no source are skipped."""
    picked: list[SliceQuery] = []
    skipped: list[str] = []
    for q in queries:
        source = source_of(q.query_image_id, index_ids)
        if source is QuerySource.SUPPLIED and q.query_image_id not in supplied:
            skipped.append(q.query_id)
            continue
        picked.append(
            SliceQuery(
                query_id=q.query_id,
                set_id=q.set_id,
                source=source,
                image_id=q.query_image_id,
                file_id=file_of.get(q.query_image_id, q.query_image_id)
                if source is QuerySource.CATALOG
                else "",
                hidden=tuple(dict.fromkeys((q.query_image_id, *q.exclude_image_ids))),
                expected=q.expected_entity_ids,
            )
        )
    return MeasureSpec(index=index, queries=tuple(picked), skipped=tuple(skipped))


def prepare(settings: Settings, s3: S3Client, index: str, index_ids: Sequence[str]) -> MeasureSpec:
    """The spec, the supplied vectors (with each photo's chroma spread) and the outside photos,
    to the images bucket."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    from miroir.eval.router3 import query_chroma  # noqa: PLC0415 (fetches photos)
    from miroir.eval.sets import load_queries  # noqa: PLC0415
    from miroir.eval.vectors import read_vectors  # noqa: PLC0415

    queries = [q for s in SETS for q in load_queries(settings, s, "tune")]
    runs = [read_vectors(settings.vectors_path(r)) for r in QUERY_RUNS]
    rows = [r.index() for r in runs]
    data = pq.read_table(settings.scaleway_dir / IMAGE_FILES, columns=["image_id", "file_id"])  # pyright: ignore[reportUnknownMemberType]
    columns = data.to_pydict()
    file_of = dict(zip(columns["image_id"], columns["file_id"], strict=True))
    spec = measure_queries(queries, set(index_ids), file_of, set(rows[0]) & set(rows[1]), index)
    logger.info("%d queries, %d skipped (no vectors)", len(spec.queries), len(spec.skipped))
    supplied = sorted({q.image_id for q in spec.queries if q.source is QuerySource.SUPPLIED})
    chroma = query_chroma(settings, supplied)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        ids=np.array(supplied),
        a7=np.stack([np.asarray(runs[0].vectors[rows[0][i]], np.float32) for i in supplied]),
        a3=np.stack([np.asarray(runs[1].vectors[rows[1][i]], np.float32) for i in supplied]),
        chroma=np.asarray([chroma[i] for i in supplied], dtype=np.float32),
    )
    bucket = bucket_name(settings, Bucket.IMAGES)
    s3.put_object(Bucket=bucket, Key=supplied_key(index), Body=buffer.getvalue())
    for q in spec.queries:
        if q.source is QuerySource.OUTSIDE:
            body = settings.outside_image_path(q.image_id).read_bytes()
            s3.put_object(Bucket=bucket, Key=outside_key(index, q.image_id), Body=body)
    s3.put_object(Bucket=bucket, Key=spec_key(index), Body=spec_json(spec))
    return spec


# --- the ranks back on the Mac -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryRanks:
    """One query's rank of its right entity, crops off and on (0: no right entity in the
    index), and the milliseconds of each step of the search with crops."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    query_id: str
    set_id: str
    bucket: str
    off: int
    on: int
    off_ms: float
    on_ms: float
    crops_ms: float
    on_few: int = 0
    """The rank with fewer crops rescored (`miroir.remote.crop_measure.FEW`)."""
    few_crops_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class MeasureOut:
    __pydantic_config__ = ConfigDict(extra="forbid")

    index: str
    gpu: str
    ranks: tuple[QueryRanks, ...]
    photo_search_ms: Mapping[str, tuple[float, ...]]
    """End to end (`Engine.search` on the photo, models included), "off" and "on"."""
    memory_gb: Mapping[str, float] = field(default_factory=dict[str, float])
    """GPU memory the search models, gallery and crops took (most allocated at once), and
    the process's host memory at its largest."""


_OUT = TypeAdapter(MeasureOut)


def out_json(out: MeasureOut) -> bytes:
    return _OUT.dump_json(out)


def read_out(data: bytes) -> MeasureOut:
    return _OUT.validate_json(data)


def summarize(out: MeasureOut) -> dict[str, dict[str, float]]:
    """Per set: queries, hit@k off and on, first place lost and gained, median times."""
    table: dict[str, dict[str, float]] = {}
    for set_id in dict.fromkeys(r.set_id for r in out.ranks):
        mine = [r for r in out.ranks if r.set_id == set_id and r.off > 0]
        row: dict[str, float] = {"queries": float(len(mine))}
        for k in HIT_KS:
            for side in ("off", "on", "on_few"):
                hit = [getattr(r, side) <= k for r in mine]
                row[f"hit@{k} {side}"] = round(100 * float(np.mean(hit)), 1) if hit else 0.0
        row["first lost"] = float(sum(r.off == 1 and r.on != 1 for r in mine))
        row["first gained"] = float(sum(r.off != 1 and r.on == 1 for r in mine))
        every = [r for r in out.ranks if r.set_id == set_id]
        row["median ms off"] = round(float(np.median([r.off_ms for r in every])), 1)
        row["median ms on"] = round(float(np.median([r.on_ms for r in every])), 1)
        row["median crops ms"] = round(float(np.median([r.crops_ms for r in every])), 1)
        row["median few crops ms"] = round(float(np.median([r.few_crops_ms for r in every])), 1)
        table[set_id] = row
    return table


def fetch_out(settings: Settings, s3: S3Client, index: str) -> Path:
    path = settings.scaleway_dir / PREFIX / f"{index}-ranks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket_name(settings, Bucket.IMAGES), out_key(index), str(path))
    return path


# --- the machine ---------------------------------------------------------------------------------


def measure_launch(settings: Settings, index: str) -> Launch:
    """`miroir scw crop-measure-run` in the job image on a GPU machine."""
    from miroir.jobs.scw_bulk import Launch  # noqa: PLC0415
    from miroir.jobs.scw_run import IMAGE_TAG, MOUNTS, REMOTE_ROOT, build_image  # noqa: PLC0415

    def start(shell: Shell, machine: Machine) -> None:
        from miroir.jobs.crop_volume import add_crop_volume, mount_point  # noqa: PLC0415
        from miroir.jobs.scaleway import InstanceApi  # noqa: PLC0415

        api = InstanceApi(settings)
        try:  # the whole crop vectors do not fit on the root volume
            add_crop_volume(
                api, shell, zone=machine.offer.zone, server_id=machine.server_id,
                mount=mount_point(f"{REMOTE_ROOT}/web-index", index), name=f"{machine.name}-crops",
            )  # fmt: skip
        finally:
            api.close()
        build_image(shell, settings)
        run = f"docker run -d --env-file {REMOTE_ROOT}/env {MOUNTS} --entrypoint miroir "
        shell.run(
            f"{run} --name watch {IMAGE_TAG} scw watch --server-id {machine.server_id} "
            f"--zone {machine.offer.zone} --deadline {machine.deadline:.0f}"
        )
        shell.run(
            f"{run} --name measure --gpus all --shm-size 8g {IMAGE_TAG} scw crop-measure-run "
            f"--index {index}"
        )

    return Launch(
        job=JOB,
        shards=(0,),
        start=start,
        alive="docker ps -q --filter name=^measure$ | wc -l",
        logs="docker logs --tail 40 measure 2>&1",
        probe="nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader "
        "&& free -g | sed -n 2p && df -h / | tail -1",
    )


def run_measure(settings: Settings, index: str, *, max_hours: float) -> Path:
    """T1, T2, T2b, T3, T5, PARTS and VET (tune halves) searched on a GPU machine with the
    site's code, crops off and on; writes eval/results/R3/SL-<index>.json, returns its path."""
    import json  # noqa: PLC0415

    from miroir.config import eval_result_path  # noqa: PLC0415
    from miroir.jobs.crop_embed import crop_offers, in_stock  # noqa: PLC0415
    from miroir.jobs.crop_slice import slice_dir  # noqa: PLC0415
    from miroir.jobs.objects import s3_client  # noqa: PLC0415
    from miroir.jobs.scaleway import InstanceApi  # noqa: PLC0415
    from miroir.jobs.scw_bulk import run_unattended  # noqa: PLC0415

    s3 = s3_client(settings)
    ids = (slice_dir(settings) / "ids.txt").read_text(encoding="utf-8").split()
    spec = prepare(settings, s3, index, ids)
    logger.info("%d queries (%d without vectors left out)", len(spec.queries), len(spec.skipped))
    api = InstanceApi(settings)
    try:
        offers = [o for o in in_stock(api, crop_offers()) if o.commercial_type != "RENDER-S"]
    finally:
        api.close()
    run_unattended(
        settings, measure_launch(settings, index), offers=offers, max_hours=max_hours,
        purpose="crop measure",
    )  # fmt: skip
    out = read_out(fetch_out(settings, s3, index).read_bytes())
    result = {
        "index": index,
        "gpu": out.gpu,
        "sets": summarize(out),
        "photo_search_ms": {k: list(v) for k, v in out.photo_search_ms.items()},
        "memory_gb": dict(out.memory_gb),
        "skipped": list(spec.skipped),
        "vet_ranks": {r.query_id: [r.off, r.on] for r in out.ranks if r.set_id == "VET"},
    }
    path = eval_result_path("R3", "SL", index)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return path
