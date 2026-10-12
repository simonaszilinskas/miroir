"""PCA to shrink MegaLoc's 8,448-d vectors for the full-sample index (technical spec §7.3).

Fitted on gallery vectors only, never on queries. A vector `x` (unit length, as stored) maps
to `normalise((x - mean) @ components)`; the website applies the same `mean` and `components`
to a query's MegaLoc vector before searching. NumPy only: no model runs here.

The fit reads the run in blocks and sums the covariance in float64, so 138,000 x 8,448 rows
never sit in memory as float32 at once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.eval.vectors import VectorSet, write_vectors

logger = logging.getLogger(__name__)

#: Rows read at once while fitting or projecting.
BLOCK_ROWS = 8192


@dataclass(frozen=True, slots=True)
class Pca:
    mean: NDArray[np.float64]
    """Mean of the gallery vectors, shape (dim,)."""
    components: NDArray[np.float32]
    """Top principal axes as columns, shape (dim, dims)."""
    explained: NDArray[np.float64]
    """Share of the gallery's variance each axis holds, shape (dims,)."""
    fitted_on: str
    """Vector run the fit read."""

    @property
    def dims(self) -> int:
        return int(self.components.shape[1])


def fit_pca(run: VectorSet, dims: int, name: str) -> Pca:
    """Principal axes of a run's vectors, from their covariance (eigh, largest first)."""
    count, dim = run.vectors.shape
    if not 0 < dims < dim:
        msg = f"cannot reduce {dim} dims to {dims}"
        raise ValueError(msg)
    total = np.zeros(dim, dtype=np.float64)
    gram = np.zeros((dim, dim), dtype=np.float64)
    for start in range(0, count, BLOCK_ROWS):
        block = np.asarray(run.vectors[start : start + BLOCK_ROWS], dtype=np.float32)
        total += block.sum(axis=0, dtype=np.float64)
        gram += (block.T @ block).astype(np.float64)
    mean = total / count
    covariance = gram / count - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(covariance)
    top = np.argsort(values)[::-1][:dims]
    explained = values[top] / values.sum()
    logger.info(
        "PCA %d -> %d on %s (%d rows): %.1f%% of variance kept",
        dim,
        dims,
        name,
        count,
        100 * explained.sum(),
    )
    return Pca(mean, vectors[:, top].astype(np.float32), explained, name)


def project(pca: Pca, vectors: NDArray[np.floating]) -> NDArray[np.float32]:
    """Centre, project and L2-normalise rows, in blocks."""
    out = np.empty((vectors.shape[0], pca.dims), dtype=np.float32)
    mean = pca.mean.astype(np.float32)
    for start in range(0, vectors.shape[0], BLOCK_ROWS):
        block = np.asarray(vectors[start : start + BLOCK_ROWS], dtype=np.float32)
        reduced = (block - mean) @ pca.components
        out[start : start + BLOCK_ROWS] = reduced / np.linalg.norm(reduced, axis=1, keepdims=True)
    return out


def write_projected(pca: Pca, run: VectorSet, out: Path) -> VectorSet:
    """Write `run` projected by `pca` as a new vector run, its notes saying how."""
    note = f"PCA {run.dim} -> {pca.dims} dims fitted on {pca.fitted_on}"
    info = replace(run.info, notes=f"{run.info.notes}; {note}" if run.info.notes else note)
    return write_vectors(out, run.ids, project(pca, run.vectors), info)


def save_pca(path: Path, pca: Pca) -> None:
    """One .npz: `mean` (float32), `components` (float32, dim x dims), `explained`, `fitted_on`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=pca.mean.astype(np.float32),
        components=pca.components,
        explained=pca.explained.astype(np.float32),
        fitted_on=np.array(pca.fitted_on),
    )
    logger.info("wrote %s", path)


def load_pca(path: Path) -> Pca:
    with np.load(path, allow_pickle=False) as data:
        return Pca(
            mean=np.asarray(data["mean"], dtype=np.float64),
            components=np.asarray(data["components"], dtype=np.float32),
            explained=np.asarray(data["explained"], dtype=np.float64),
            fitted_on=str(data["fitted_on"]),
        )
