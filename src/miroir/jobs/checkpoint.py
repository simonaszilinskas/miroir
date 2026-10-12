"""Save a paid embedding run as it goes, so a stopped run resumes without paying twice.

A checkpoint is a folder (`Settings.checkpoint_path(run)`) with two files that only grow:
`vectors.f16` (raw float16 rows, unit length) and `ids.txt` (one id per row). Rows are
written before their ids, so after a crash `ids.txt` says how many rows are whole; any
extra bytes are cut off when the checkpoint is opened again.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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
        if vectors.shape != (len(ids), vectors.shape[1]):
            msg = f"{len(ids)} ids for vectors of shape {vectors.shape}"
            raise ValueError(msg)
        if self._dim is None:
            self._dim = int(vectors.shape[1])
            self._dir.mkdir(parents=True, exist_ok=True)
            (self._dir / "dim").write_text(f"{self._dim}\n", encoding="utf-8")
        elif vectors.shape[1] != self._dim:
            msg = f"vector size {vectors.shape[1]}, checkpoint holds {self._dim}"
            raise ValueError(msg)
        with (self._dir / _VECTORS).open("ab") as out:
            out.write(np.ascontiguousarray(vectors, dtype=np.float16).tobytes())
            out.flush()
        with (self._dir / _IDS).open("a", encoding="utf-8") as out:
            out.write("".join(f"{i}\n" for i in ids))
            out.flush()
        self._ids.extend(ids)

    def vectors(self) -> NDArray[np.float16]:
        if self._dim is None:
            return np.zeros((0, 0), dtype=np.float16)
        raw = np.fromfile(self._dir / _VECTORS, dtype=np.float16)
        return raw[: len(self._ids) * self._dim].reshape(len(self._ids), self._dim)

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
        self._ids = ids_file.read_text(encoding="utf-8").splitlines() if ids_file.exists() else []
        whole = len(self._ids) * self._dim * _ROW_BYTES
        vectors = self._dir / _VECTORS
        size = vectors.stat().st_size if vectors.exists() else 0
        if size < whole:
            # Ids are written after their rows, so this means the files were edited by hand.
            msg = f"{self._dir}: fewer vector rows than ids; delete the checkpoint"
            raise ValueError(msg)
        if size > whole:
            with vectors.open("r+b") as out:
                out.truncate(whole)
            logger.warning("%s: cut %d bytes of a half-written chunk", self._dir, size - whole)
        logger.info("%s: resuming with %d vectors already paid for", self._dir, len(self._ids))
