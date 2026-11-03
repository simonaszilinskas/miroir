"""The `pop-all-crops` bundle: `pop-all` with the crop vectors of every stored file.

Built on the Mac, which only moves numbers (no model):

1. `copy_bundle` copies every object of `pop-all` to the new prefix inside the index bucket
   (a server-side copy: nothing comes down).
2. `write_crop_files` reads the coordinate pieces of the crop job (`miroir.jobs.crop_embed`,
   already downloaded), keeps the first `dims` PCA axes of each crop, renormalised, in int8
   with one scale per crop, and writes the files `miroir.web.crops` reads, image row by image
   row of `pop-all`; they go up beside the copied files.
3. `copy_whole` copies the job's whole-vector pieces into the bundle, bucket to bucket.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from miroir.jobs.objects import listing
from miroir.parallel import bounded_map
from miroir.web.crops import (
    COORDS_FILE,
    PROJECTION_FILE,
    ROW_FILE,
    RULE_FILE,
    SCALE_FILE,
    WHOLE_DIR,
    WHOLE_INDEX_FILE,
    WHOLE_SCALE_FILE,
    CropRule,
    write_rule,
    write_whole_index,
)
from miroir.web.store import quantize_rows

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

SOURCE = "pop-all"
NAME = "pop-all-crops"


def copy_bundle(s3: S3Client, bucket: str, source: str, target: str) -> int:
    """Copy `source/*` to `target/*` inside the bucket (those already there at the same size
    are kept). Returns how many were copied."""
    there = {s.key: s.size for s in listing(s3, bucket, f"{target}/")}
    todo = [
        s
        for s in listing(s3, bucket, f"{source}/")
        if there.get(f"{target}/{s.key.removeprefix(f'{source}/')}") != s.size
    ]

    def copy(key: str) -> None:
        name = key.removeprefix(f"{source}/")
        s3.copy({"Bucket": bucket, "Key": key}, bucket, f"{target}/{name}")

    list(bounded_map(copy, [s.key for s in todo], workers=8))
    logger.info("copied %d of %s to %s", len(todo), source, target)
    return len(todo)


def piece_paths(folder: Path) -> list[Path]:
    return sorted((folder / "pieces").glob("*.npz"))


def write_crop_files(
    pieces: Sequence[Path],
    out: Path,
    *,
    index_ids: Sequence[str],
    file_of: Mapping[str, str],
    projection: Path,
    dims: int,
    rule: CropRule,
) -> dict[str, int]:
    """Write the crop files of `miroir.web.crops` into `out`, all but the whole vectors
    (`copy_whole` puts those in the bucket). Returns counts."""
    out.mkdir(parents=True, exist_ok=True)
    counts: list[int] = []
    for path in pieces:
        with np.load(path, allow_pickle=False) as data:
            counts.append(len(data["files"]) * rule.per_file)
            if len(data["files"]) and int(data["per_file"]) != rule.per_file:
                msg = (
                    f"{path}: {int(data['per_file'])} crops per file, the rule says {rule.per_file}"
                )
                raise ValueError(msg)
    total = sum(counts)
    coords = np.lib.format.open_memmap(
        out / COORDS_FILE, mode="w+", dtype=np.int8, shape=(total, dims)
    )
    scale = np.empty(total, dtype=np.float32)
    whole_scale = np.empty(total, dtype=np.float32)
    file_row: dict[str, int] = {}
    failed = 0
    at = 0
    for path, count in zip(pieces, counts, strict=True):
        with np.load(path, allow_pickle=False) as data:
            files = [str(f) for f in data["files"].tolist()]
            failed += len(data["failed"])
            if count:
                short = np.asarray(data["coords"][:, :dims], dtype=np.float32)
                short /= np.linalg.norm(short, axis=1, keepdims=True)
                q, s = quantize_rows(short)
                coords[at : at + count] = q
                scale[at : at + count] = s
                whole_scale[at : at + count] = data["whole_scale"]
        for k, f in enumerate(files):
            file_row[f] = at // rule.per_file + k
        at += count
    coords.flush()
    del coords
    np.save(out / SCALE_FILE, scale)
    np.save(out / WHOLE_SCALE_FILE, whole_scale)
    write_whole_index(out, [p.stem for p in pieces], counts)
    row_file = np.fromiter(
        (file_row.get(file_of.get(i, ""), -1) for i in index_ids), np.int32, len(index_ids)
    )
    np.save(out / ROW_FILE, row_file)
    with np.load(projection, allow_pickle=False) as data:
        np.savez(
            out / PROJECTION_FILE,
            mean=np.asarray(data["mean"], np.float32),
            components=np.asarray(data["components"], np.float32)[:dims],
        )
    write_rule(out, rule)
    facts = {
        "crop_files": len(file_row),
        "crops": total,
        "failed_files": failed,
        "rows": len(index_ids),
        "rows_with_crops": int((row_file >= 0).sum()),
    }
    (out / "crops-facts.json").write_text(json.dumps(facts, indent=1) + "\n")
    logger.info("crop files: %s", facts)
    return facts


def copy_whole(s3: S3Client, bucket: str, job: str, index_bucket: str, target: str) -> int:
    """Copy the crop job's whole-vector pieces into the bundle (`crops_whole/`), bucket to
    bucket: nothing comes down."""
    from miroir.jobs.crop_embed import PREFIX  # noqa: PLC0415

    source = f"{PREFIX}/{job}/whole/"
    there = {s.key: s.size for s in listing(s3, index_bucket, f"{target}/{WHOLE_DIR}/")}
    todo = [
        s
        for s in listing(s3, bucket, source)
        if there.get(f"{target}/{WHOLE_DIR}/{s.key.removeprefix(source)}") != s.size
    ]

    def copy(key: str) -> None:
        name = key.removeprefix(source)
        s3.copy({"Bucket": bucket, "Key": key}, index_bucket, f"{target}/{WHOLE_DIR}/{name}")

    list(bounded_map(copy, [s.key for s in todo], workers=16))
    logger.info("copied %d whole-vector pieces into %s", len(todo), target)
    return len(todo)


def crop_file_paths(folder: Path) -> list[Path]:
    names = (
        COORDS_FILE, SCALE_FILE, ROW_FILE, PROJECTION_FILE, WHOLE_INDEX_FILE, WHOLE_SCALE_FILE,
        RULE_FILE, "crops-facts.json",
    )  # fmt: skip
    return [folder / n for n in names]
