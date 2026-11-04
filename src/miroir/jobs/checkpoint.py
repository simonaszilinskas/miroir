"""Save a paid embedding run as it goes, so a stopped run resumes without paying twice.

A checkpoint is a folder (`Settings.checkpoint_path(run)`) with two files that only grow:
`vectors.f16` (raw float16 rows, unit length) and `ids.txt` (one id per row), both on disk
(`miroir.jobs.durable`) before `append` returns. Rows are written before their ids. When the
checkpoint is opened again after a crash, a half-written id line is dropped, rows past the
last whole id are cut off, and ids past the last whole row are dropped: what is kept is
always whole, and what is dropped is asked again.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.jobs.durable import (
    append_bytes,
    append_lines,
    read_lines,
    rewrite_lines,
    truncate,
    write_atomic,
)

logger = logging.getLogger(__name__)

_VECTORS = "vectors.f16"
_IDS = "ids.txt"
_ROW_BYTES = 2  # float16


class Checkpoint:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dim: int | None = None
        self._ids: list[str] = []
        self._load()

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids)

    def append(self, ids: list[str], vectors: NDArray[np.float16]) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            msg = f"{len(ids)} ids for vectors of shape {vectors.shape}"
            raise ValueError(msg)
        if any("\n" in i for i in ids):
            msg = "an id holds a newline"
            raise ValueError(msg)
        if self._dim is None:
            self._dim = int(vectors.shape[1])
            write_atomic(self._dir / "dim", f"{self._dim}\n".encode())
        elif vectors.shape[1] != self._dim:
            msg = f"vector size {vectors.shape[1]}, checkpoint holds {self._dim}"
            raise ValueError(msg)
        append_bytes(self._dir / _VECTORS, np.ascontiguousarray(vectors, np.float16).tobytes())
        append_lines(self._dir / _IDS, (i.encode() for i in ids))
        self._ids.extend(ids)

    def vectors(self) -> NDArray[np.float16]:
        if self._dim is None:
            return np.zeros((0, 0), dtype=np.float16)
        raw = np.fromfile(self._dir / _VECTORS, dtype=np.float16)
        return raw[: len(self._ids) * self._dim].reshape(len(self._ids), self._dim)

    def only(self, wanted: Iterable[str]) -> tuple[tuple[str, ...], NDArray[np.float16]]:
        """The ids it holds that are in `wanted`, once each, with their vectors."""
        keep = set(wanted)
        rows: dict[str, int] = {}
        for n, i in enumerate(self._ids):
            if i in keep:
                rows.setdefault(i, n)
        dropped = len(self._ids) - len(rows)
        if dropped:
            logger.warning("%s: left out %d saved vector(s) not asked for", self._dir, dropped)
        return tuple(rows), self.vectors()[list(rows.values())]

    def remove(self) -> None:
        for name in (_VECTORS, _IDS, "dim"):
            (self._dir / name).unlink(missing_ok=True)
        if self._dir.exists():
            self._dir.rmdir()

    def _load(self) -> None:
        dim_file = self._dir / "dim"
        if not dim_file.exists():
            return
        self._dim = int(dim_file.read_text(encoding="utf-8"))
        ids_file = self._dir / _IDS
        self._ids = [line.decode() for line in read_lines(ids_file)]
        vectors = self._dir / _VECTORS
        size = vectors.stat().st_size if vectors.exists() else 0
        rows = size // (self._dim * _ROW_BYTES)
        if rows < len(self._ids):
            # A crash between the rows and their ids cannot do this; a disk that lost the
            # end of the rows can. Keep what is whole; the rest is asked again.
            logger.warning(
                "%s: %d ids but %d whole rows; the last %d will be asked again",
                self._dir,
                len(self._ids),
                rows,
                len(self._ids) - rows,
            )
            self._ids = self._ids[:rows]
            rewrite_lines(ids_file, (i.encode() for i in self._ids))
        whole = len(self._ids) * self._dim * _ROW_BYTES
        if size > whole:
            truncate(vectors, whole)
            logger.warning("%s: cut %d bytes of a half-written chunk", self._dir, size - whole)
        logger.info("%s: resuming with %d vectors already paid for", self._dir, len(self._ids))
