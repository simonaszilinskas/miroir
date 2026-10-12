"""The vector contract between embedding runs and the scoring harness (technical spec §7.1).

One embedding run is one folder, `config.Settings.vectors_path(run)`, holding three files:

- `vectors.npy`: float16, shape (count, dim), C order, every row L2-normalised.
- `ids.txt`: `count` lines, UTF-8, one id per line, row i of `vectors.npy` is line i.
  POP ids are `image_id`s from `images.parquet`; benchmark ids are any unique string
  without whitespace.
- `meta.json`: what the run was (`RunInfo`) plus `format_version`, `dtype`, `dim`, `count`.

Write with `write_vectors`, read with `read_vectors`. Both check every rule above, so a run
that loads is a run the harness can score.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter, ValidationError

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
VECTORS_FILE = "vectors.npy"
IDS_FILE = "ids.txt"
META_FILE = "meta.json"

#: Largest allowed distance of a stored row's norm from 1. float16 rounding stays under 1e-3.
NORM_TOLERANCE = 5e-3

_ID_RE = re.compile(r"^\S+$")
_CHUNK_ROWS = 65_536

Vectors = NDArray[np.float16]


class Route(StrEnum):
    """Where an embedding run computed its vectors. Never the Mac: no model runs there."""

    OPENROUTER = "openrouter"
    HOSTED = "hosted"
    """Any other hosted model service; name it in `RunInfo.notes`."""
    HF_JOBS = "hf-jobs"


@dataclass(frozen=True, slots=True)
class RunInfo:
    """What one embedding run was. Stored in `meta.json` with the shape fields."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    option: str
    """Option id from experiments.md §2, e.g. "A1"."""
    model: str
    """Exact model id, e.g. "facebook/dinov2-giant" or "google/gemini-embedding-2"."""
    resolution: int
    """Pixels on the side the model saw: crop size, or longest side sent to an API."""
    pooling: str
    """How one vector came out of the model: "cls", "gem", "vlad", or "api" for hosted ones."""
    gallery: str
    """Which image list was embedded: "screening-20k", "full", "amstertime"..."""
    created: date
    cost_usd: float
    wall_time_s: float
    route: Route
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("option", "model", "pooling", "gallery"):
            if not getattr(self, name):
                msg = f"RunInfo.{name} is empty"
                raise ValueError(msg)
        if self.resolution <= 0 or self.cost_usd < 0 or self.wall_time_s < 0:
            msg = f"RunInfo has a negative or zero size, cost or time: {self}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class VectorMeta:
    """The whole of `meta.json`."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    format_version: int
    dtype: Literal["float16"]
    dim: int
    count: int
    info: RunInfo


_META = TypeAdapter(VectorMeta)


@dataclass(frozen=True, slots=True)
class VectorSet:
    """One loaded run. `vectors[i]` is the vector of `ids[i]`."""

    ids: tuple[str, ...]
    vectors: Vectors
    info: RunInfo

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def index(self) -> dict[str, int]:
        """Row number of each id."""
        return {image_id: i for i, image_id in enumerate(self.ids)}


class VectorFormatError(ValueError):
    """A run folder breaks the vector contract."""


def write_vectors(
    directory: Path, ids: Sequence[str], vectors: NDArray[np.floating], info: RunInfo
) -> VectorSet:
    """Normalise, cast to float16 and write one run. Refuses to overwrite a finished run."""
    if (directory / META_FILE).exists():
        msg = f"{directory} already holds a run; pick a new run name"
        raise FileExistsError(msg)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0:
        msg = f"vectors shape {matrix.shape} does not match {len(ids)} ids"
        raise VectorFormatError(msg)
    _check_ids(tuple(ids))
    if not np.isfinite(matrix).all():
        msg = "vectors hold NaN or infinite values"
        raise VectorFormatError(msg)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        msg = f"{int((norms == 0).sum())} zero vector(s): cannot normalise"
        raise VectorFormatError(msg)
    stored = (matrix / norms).astype(np.float16)
    _check_rows(stored)

    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / VECTORS_FILE, stored, allow_pickle=False)
    (directory / IDS_FILE).write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    meta = VectorMeta(
        format_version=FORMAT_VERSION,
        dtype="float16",
        dim=int(stored.shape[1]),
        count=int(stored.shape[0]),
        info=info,
    )
    # meta.json goes last: its presence marks the run as complete.
    (directory / META_FILE).write_bytes(_META.dump_json(meta, indent=2) + b"\n")
    logger.info("wrote %d vectors of dim %d to %s", meta.count, meta.dim, directory)
    return VectorSet(ids=tuple(ids), vectors=stored, info=info)


def read_vectors(directory: Path, *, mmap: bool = True) -> VectorSet:
    """Load one run and check it against the contract. Raises VectorFormatError if it breaks it."""
    meta = read_meta(directory)
    try:
        loaded = np.load(
            directory / VECTORS_FILE, mmap_mode="r" if mmap else None, allow_pickle=False
        )
    except (OSError, ValueError) as err:
        msg = f"{directory / VECTORS_FILE}: {err}"
        raise VectorFormatError(msg) from err
    if loaded.dtype != np.float16 or loaded.shape != (meta.count, meta.dim):
        msg = (
            f"{directory / VECTORS_FILE}: {loaded.dtype} {loaded.shape}, "
            f"meta.json says float16 ({meta.count}, {meta.dim})"
        )
        raise VectorFormatError(msg)
    vectors: Vectors = loaded
    ids = tuple((directory / IDS_FILE).read_text(encoding="utf-8").splitlines())
    if len(ids) != meta.count:
        msg = f"{directory / IDS_FILE}: {len(ids)} ids, meta.json says {meta.count}"
        raise VectorFormatError(msg)
    _check_ids(ids)
    _check_rows(vectors)
    return VectorSet(ids=ids, vectors=vectors, info=meta.info)


def read_meta(directory: Path) -> VectorMeta:
    path = directory / META_FILE
    if not path.exists():
        msg = f"{path} is missing: the run is unfinished or not a vector run"
        raise VectorFormatError(msg)
    try:
        meta = _META.validate_json(path.read_bytes())
    except ValidationError as err:
        msg = f"{path}: {err}"
        raise VectorFormatError(msg) from err
    if meta.format_version != FORMAT_VERSION:
        msg = f"{path}: format_version {meta.format_version}, expected {FORMAT_VERSION}"
        raise VectorFormatError(msg)
    return meta


def merge_runs(directory: Path, runs: Sequence[VectorSet], *, gallery: str) -> VectorSet:
    """Write one run holding the rows of several runs of the same model, e.g. a gallery run
    and a run of the images a larger gallery adds, so nothing is embedded twice.

    Refuses runs of different models, sizes or pooling, and an id found in two runs.
    Cost and time are the sums of the parts.
    """
    if len(runs) < 2:
        msg = "a merge needs two or more runs"
        raise ValueError(msg)
    first = runs[0].info
    same = (first.option, first.model, first.resolution, first.pooling, runs[0].dim)
    for run in runs[1:]:
        info = run.info
        if (info.option, info.model, info.resolution, info.pooling, run.dim) != same:
            msg = f"cannot merge {info.model} {info.pooling} @{info.resolution} into {same}"
            raise ValueError(msg)
    ids = [i for run in runs for i in run.ids]
    vectors = np.concatenate([np.asarray(run.vectors) for run in runs])
    parts = ", ".join(f"{run.info.gallery} ({len(run.ids)})" for run in runs)
    notes = "; ".join(n for n in (f"merged from {parts}", *(r.info.notes for r in runs)) if n)
    info = RunInfo(
        option=first.option,
        model=first.model,
        resolution=first.resolution,
        pooling=first.pooling,
        gallery=gallery,
        created=max(run.info.created for run in runs),
        cost_usd=sum(run.info.cost_usd for run in runs),
        wall_time_s=sum(run.info.wall_time_s for run in runs),
        route=first.route,
        notes=notes,
    )
    return write_vectors(directory, ids, vectors, info)


def meta_dict(info: RunInfo) -> dict[str, object]:
    """`RunInfo` as plain JSON values, for copying into result files."""
    return json.loads(TypeAdapter(RunInfo).dump_json(info))


def _check_ids(ids: tuple[str, ...]) -> None:
    bad = [i for i in ids if not _ID_RE.match(i)]
    if bad:
        msg = f"{len(bad)} empty id(s) or id(s) with whitespace, e.g. {bad[0]!r}"
        raise VectorFormatError(msg)
    if len(set(ids)) != len(ids):
        msg = f"{len(ids) - len(set(ids))} duplicate id(s)"
        raise VectorFormatError(msg)


def _check_rows(vectors: Vectors) -> None:
    """Every row finite and of norm 1 within NORM_TOLERANCE. Reads in chunks to bound memory."""
    for start in range(0, vectors.shape[0], _CHUNK_ROWS):
        chunk = np.asarray(vectors[start : start + _CHUNK_ROWS], dtype=np.float32)
        if not np.isfinite(chunk).all():
            msg = f"rows {start}..{start + len(chunk)} hold NaN or infinite values"
            raise VectorFormatError(msg)
        off = np.abs(np.linalg.norm(chunk, axis=1) - 1.0)
        if (off > NORM_TOLERANCE).any():
            row = start + int(off.argmax())
            msg = f"row {row} has norm off by {float(off.max()):.4f}: not L2-normalised"
            raise VectorFormatError(msg)
