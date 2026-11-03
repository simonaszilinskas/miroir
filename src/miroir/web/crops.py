"""Crop vectors in an index bundle: a photo of a part finds the whole it stands in.

Each stored POP file is cut into a 3x3 grid of overlapping windows (`miroir.jobs.crop_embed`),
each embedded with MegaLoc and kept short (`ProjectedCrops`): the first `dims` axes of a PCA,
in int8 with one scale per crop, and the crop's dot with the PCA's mean. A query's MegaLoc
vector meets every crop at once (`CropStore.best_per_file`); each image row takes the best
crop of its file (`row_scores`). The `CropRule.shortlist` best rows join the MegaLoc
shortlist (`miroir.web.store`), and there an image's MegaLoc score becomes
max(whole, weight * best crop): one number per image, so routing and fusion are unchanged.

| File | Holds |
|---|---|
| `crops_coords.npy` | int8 (crops, dims): the crops' PCA coordinates, file by file |
| `crops_scale.npy` | float32 (crops,): each row's scale |
| `crops_mean_dot.npy` | float32 (crops,): each crop's dot with the PCA mean |
| `crops_row_file.npy` | int32 (image rows,): each image row's crop file, -1 for none |
| `crops_projection.npz` | `mean` (8,448) and `components` (dims, 8,448) |
| `crops.json` | `CropRule` |
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]
Ints = NDArray[np.int64]

COORDS_FILE = "crops_coords.npy"
SCALE_FILE = "crops_scale.npy"
MEAN_DOT_FILE = "crops_mean_dot.npy"
ROW_FILE = "crops_row_file.npy"
PROJECTION_FILE = "crops_projection.npz"
RULE_FILE = "crops.json"
#: Crop rows scored at once on the CPU.
BLOCK_ROWS = 1 << 20


@dataclass(frozen=True, slots=True)
class CropRule:
    per_file: int = 9
    """Windows per file (a 3x3 grid)."""
    shortlist: int = 2000
    """Image rows, by best crop, that join the MegaLoc shortlist."""
    weight: float = 1.0
    """An image scores max(whole, weight * best crop)."""


@dataclass(frozen=True, slots=True)
class CropQuery:
    """A query's side of `ProjectedCrops`: C (q - m), and m . q - m . m."""

    coords: Floats
    offset: float


class CropStore(Protocol):
    def best_per_file(self, query: CropQuery) -> Floats:
        """Each crop file's best crop score (approximate whole cosine)."""
        ...


@dataclass(frozen=True, slots=True)
class NumpyCrops:
    """The crops in host memory, scored on the CPU in blocks (the Mac's tests, and the demo
    machine without a GPU)."""

    coords: NDArray[np.int8]
    scale: Floats
    mean_dot: Floats
    per_file: int

    def best_per_file(self, query: CropQuery) -> Floats:
        out = np.empty(len(self.coords) // self.per_file, dtype=np.float32)
        step = BLOCK_ROWS - BLOCK_ROWS % self.per_file
        for start in range(0, len(self.coords), step):
            stop = min(start + step, len(self.coords))
            block = np.asarray(self.coords[start:stop], dtype=np.float32)
            sims = (block @ query.coords) * self.scale[start:stop] + self.mean_dot[start:stop]
            at = start // self.per_file
            out[at : at + (stop - start) // self.per_file] = sims.reshape(-1, self.per_file).max(
                axis=1
            )
        return out + np.float32(query.offset)


@dataclass(frozen=True, slots=True)
class CropIndex:
    rule: CropRule
    mean: Floats
    components: Floats
    row_file: NDArray[np.int32]
    store: CropStore

    def query(self, a3_full: Floats) -> CropQuery:
        """A unit MegaLoc query vector (whole, 8,448 numbers) on the crops' axes."""
        coords = (a3_full - self.mean) @ self.components.T
        return CropQuery(
            coords.astype(np.float32), float(a3_full @ self.mean - self.mean @ self.mean)
        )

    def row_scores(self, a3_full: Floats, mask: NDArray[np.bool_]) -> Floats:
        """Each image row's best crop score; -inf for a row with no crops or filtered out."""
        best = self.store.best_per_file(self.query(a3_full))
        has = (self.row_file >= 0) & mask
        out = np.full(len(self.row_file), -np.inf, dtype=np.float32)
        out[has] = best[self.row_file[has]]
        return out


def has_crops(directory: Path) -> bool:
    return (directory / RULE_FILE).exists()


def read_rule(directory: Path) -> CropRule:
    return CropRule(**json.loads((directory / RULE_FILE).read_text()))


def write_rule(directory: Path, rule: CropRule) -> None:
    (directory / RULE_FILE).write_text(json.dumps(asdict(rule), indent=1) + "\n")


def read_projection(directory: Path) -> tuple[Floats, Floats]:
    with np.load(directory / PROJECTION_FILE, allow_pickle=False) as data:
        return (
            np.asarray(data["mean"], dtype=np.float32),
            np.asarray(data["components"], dtype=np.float32),
        )


def read_numpy_crops(directory: Path, *, mmap: bool = False) -> NumpyCrops:
    rule = read_rule(directory)
    coords = np.load(directory / COORDS_FILE, mmap_mode="r" if mmap else None)
    return NumpyCrops(
        coords=coords,
        scale=np.load(directory / SCALE_FILE).astype(np.float32),
        mean_dot=np.load(directory / MEAN_DOT_FILE).astype(np.float32),
        per_file=rule.per_file,
    )


def read_crops(directory: Path, store: CropStore | None = None) -> CropIndex | None:
    """The bundle's crops, scored by `store` (host memory and the CPU by default); None for
    a bundle without crops."""
    if not has_crops(directory):
        return None
    rule = read_rule(directory)
    mean, components = read_projection(directory)
    row_file = np.load(directory / ROW_FILE).astype(np.int32)
    store = store or read_numpy_crops(directory)
    logger.info(
        "crops: %d axes, %d image rows with crops, shortlist %d, weight %.2f",
        components.shape[0],
        int((row_file >= 0).sum()),
        rule.shortlist,
        rule.weight,
    )
    return CropIndex(rule, mean, components, row_file, store)
