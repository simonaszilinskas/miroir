"""MegaLoc stored small enough for all of POP: a shortlist on the GPU, exact cosines from RAM.

At 3 million images MegaLoc's 8,448-d vectors weigh 51 GB in float16: more than any one
mid-range GPU holds. A bundle built with a shortlist (`IndexSpec.a3_shortlist_dims`) keeps:

| File | Sits in | Holds |
|---|---|---|
| `a3.npy`, `a3_projection.npz` | GPU | MegaLoc reduced by PCA (float16), as a PCA bundle |
| `a3_exact.npy`, `a3_exact_scale.npy` | host memory | every 8,448-d vector in int8, row scales |

A photo search takes the `SHORTLIST` images closest by EmbeddingGemma 2 and the `SHORTLIST`
closest by reduced MegaLoc (filters applied first), then reads their int8 rows and scores them
exactly. Every other image scores -inf for MegaLoc, so a MegaLoc-weighted search ranks only
shortlisted images. On the full-f2 tune halves this keeps every set within a point of the
float16 search (docs/results.md, "Storage for all of POP").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]
Ints = NDArray[np.int64]

EXACT_FILE = "a3_exact.npy"
EXACT_SCALE_FILE = "a3_exact_scale.npy"
#: Images each model puts on the shortlist (so at most twice this many are scored exactly).
SHORTLIST = 2000
#: Rows quantised at once.
BLOCK_ROWS = 8192


@dataclass(frozen=True, slots=True)
class ExactStore:
    """Unit vectors in int8 with one float32 scale per row: row * scale is the vector."""

    rows: NDArray[np.int8]
    scale: Floats

    def cosines(self, query: Floats, rows: Ints) -> Floats:
        """Cosine of a unit query with the given rows."""
        block = np.asarray(self.rows[rows], dtype=np.float32)
        return (block @ query) * self.scale[rows]


def quantize_rows(
    matrix: NDArray[np.floating], rows: Ints | None = None
) -> tuple[NDArray[np.int8], Floats]:
    """Each row (of `rows`, in that order, or all) to int8, scaled so its largest entry is
    127. Read in blocks, so a memory-mapped matrix never sits in memory whole."""
    order: Ints = np.arange(matrix.shape[0], dtype=np.int64) if rows is None else rows
    count = len(order)
    out = np.empty((count, matrix.shape[1]), dtype=np.int8)
    scale = np.empty(count, dtype=np.float32)
    for start in range(0, count, BLOCK_ROWS):
        block = np.asarray(matrix[order[start : start + BLOCK_ROWS]], dtype=np.float32)
        step = np.abs(block).max(axis=1) / 127.0
        step[step == 0] = 1.0
        out[start : start + len(block)] = np.clip(np.rint(block / step[:, None]), -127, 127)
        scale[start : start + len(block)] = step
    return out, scale


def write_exact(directory: Path, matrix: NDArray[np.floating], rows: Ints | None = None) -> None:
    quantized, scale = quantize_rows(matrix, rows)
    np.save(directory / EXACT_FILE, quantized, allow_pickle=False)
    np.save(directory / EXACT_SCALE_FILE, scale, allow_pickle=False)


def read_exact(directory: Path, *, mmap: bool = False) -> ExactStore | None:
    """The bundle's int8 store, or None for a bundle that keeps MegaLoc whole on the GPU."""
    path = directory / EXACT_FILE
    if not path.exists():
        return None
    rows = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    scale = np.load(directory / EXACT_SCALE_FILE, allow_pickle=False)
    if rows.dtype != np.int8 or scale.shape != (rows.shape[0],):
        msg = f"{path}: int8 rows with one scale each expected, got {rows.dtype} {rows.shape}"
        raise ValueError(msg)
    logger.info("exact MegaLoc store: %s int8 rows, %.1f GB", rows.shape, rows.nbytes / 1e9)
    return ExactStore(rows=rows, scale=np.asarray(scale, dtype=np.float32))


def top_rows(scores: Floats, size: int) -> Ints:
    """The `size` best finite rows, in no order."""
    alive = np.flatnonzero(np.isfinite(scores))
    if alive.size <= size:
        return alive.astype(np.int64)
    best = np.argpartition(-scores[alive], size)[:size]
    return alive[best].astype(np.int64)


def shortlist(a7: Floats, reduced: Floats, size: int = SHORTLIST) -> Ints:
    """Rows on either model's shortlist (filtered images score -inf and stay off)."""
    return np.union1d(top_rows(a7, size), top_rows(reduced, size)).astype(np.int64)


def shortlist_cosines(
    store: ExactStore, query: Floats, a7: Floats, reduced: Floats, size: int = SHORTLIST
) -> Floats:
    """Exact MegaLoc cosines of the shortlisted images; -inf for every other image."""
    rows = shortlist(a7, reduced, size)
    out = np.full(a7.shape, -np.inf, dtype=np.float32)
    out[rows] = store.cosines(query, rows)
    return out
