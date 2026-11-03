"""Crop vectors in an index bundle: a photo of a part finds the whole it stands in.

Each stored POP file is cut into a 3x3 grid of overlapping windows (`miroir.jobs.crop_embed`)
and each window embedded with MegaLoc. A search uses them in two stages, as the crop slice
measured (docs/results.md, "Short crops"): short vectors alone let look-alike crops pass the
right answer, short vectors to pick and whole ones to score do not.

1. **Pick.** Every crop is kept short: its first `dims` PCA axes, renormalised, in int8 with
   one scale per crop (on the GPU, `CropStore.top_crops`). The query meets all of them; the
   `CropRule.rescore` best crops go on.
2. **Score.** Those crops' whole vectors (8,448 numbers in int8, memory-mapped from disk,
   `WholeCrops`) give their exact cosine. Each image row takes the best rescored crop of its
   file (`row_scores`).

The `CropRule.shortlist` best rows join the MegaLoc shortlist (`miroir.web.store`); there an
image's MegaLoc score becomes max(whole, weight * best crop): one number per image, so routing
and fusion are unchanged.

| File | Holds |
|---|---|
| `crops_coords.npy` | int8 (crops, dims): unit PCA coordinates, file by file, 9 crops each |
| `crops_scale.npy` | float32 (crops,): each row's int8 scale |
| `crops_row_file.npy` | int32 (image rows,): each image row's crop file, -1 for none |
| `crops_projection.npz` | `mean` (8,448) and `components` (dims, 8,448) |
| `crops_whole/<piece>.npy` | int8 (crops of the piece, 8,448): the whole vectors |
| `crops_whole.json` | each piece's name and first crop row; the whole rows' scales are in |
| `crops_whole_scale.npy` | float32 (crops,) |
| `crops.json` | `CropRule` |
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from miroir.parallel import bounded_map

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]
Ints = NDArray[np.int64]

COORDS_FILE = "crops_coords.npy"
SCALE_FILE = "crops_scale.npy"
ROW_FILE = "crops_row_file.npy"
PROJECTION_FILE = "crops_projection.npz"
WHOLE_DIR = "crops_whole"
WHOLE_INDEX_FILE = "crops_whole.json"
WHOLE_SCALE_FILE = "crops_whole_scale.npy"
RULE_FILE = "crops.json"
#: Crop rows scored at once on the CPU.
BLOCK_ROWS = 1 << 20
#: Reads of whole crop vectors at once.
READERS = 64


@dataclass(frozen=True, slots=True)
class CropRule:
    per_file: int = 9
    """Windows per file (a 3x3 grid)."""
    rescore: int = 5_000
    """Crops picked by their short vectors and scored whole, per search."""
    shortlist: int = 2000
    """Image rows, by best crop, that join the MegaLoc shortlist."""
    weight: float = 1.0
    """An image scores max(whole, weight * best crop)."""


class CropStore(Protocol):
    def top_crops(self, query: Floats, count: int) -> Ints:
        """The `count` crop rows whose short vectors best meet the unit short query."""
        ...


@dataclass(frozen=True, slots=True)
class NumpyCrops:
    """The short crops in host memory, picked on the CPU in blocks (the Mac's tests, and the
    demo machine without a GPU)."""

    coords: NDArray[np.int8]
    scale: Floats

    def top_crops(self, query: Floats, count: int) -> Ints:
        rows: list[Ints] = []
        sims: list[Floats] = []
        for start in range(0, len(self.coords), BLOCK_ROWS):
            stop = min(start + BLOCK_ROWS, len(self.coords))
            block = np.asarray(self.coords[start:stop], dtype=np.float32) @ query
            block *= self.scale[start:stop]
            keep = min(count, len(block))
            best = np.argpartition(-block, keep - 1)[:keep]
            rows.append(best.astype(np.int64) + start)
            sims.append(block[best])
        every, scores = np.concatenate(rows), np.concatenate(sims)
        keep = min(count, len(every))
        return np.sort(every[np.argpartition(-scores, keep - 1)[:keep]])


@dataclass(frozen=True, slots=True)
class _Piece:
    path: Path
    offset: int
    """Where the rows start in the `.npy` file (past its header)."""
    width: int


@dataclass(frozen=True, slots=True)
class WholeCrops:
    """The crops' whole vectors in int8, one `.npy` file per piece of the crop job; only the
    rows a search rescores are read, each run of neighbouring rows in one read, `READERS`
    reads at once: a network disk answers each read in about a millisecond, but serves many
    at a time. Each read opens its file and closes it (a process may hold only about 1,000
    files open, and there are 4,096 pieces)."""

    folder: Path
    starts: Ints
    """First crop row of each piece, in order."""
    names: tuple[str, ...]
    scale: Floats
    _pieces: dict[int, _Piece] = field(default_factory=dict[int, _Piece])

    def _piece(self, k: int) -> _Piece:
        if k not in self._pieces:
            path = self.folder / f"{self.names[k]}.npy"
            with path.open("rb") as head:
                version = np.lib.format.read_magic(head)
                read = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _, dtype = read(head)
                offset = head.tell()
            if dtype != np.int8:
                msg = f"{path}: int8 rows expected, got {dtype}"
                raise ValueError(msg)
            self._pieces[k] = _Piece(path, offset, int(shape[1]))
        return self._pieces[k]

    def rows(self, rows: Ints) -> NDArray[np.int8]:
        """These crop rows, in that order."""
        order = np.argsort(rows, kind="stable")
        ordered = rows[order]
        pieces = np.searchsorted(self.starts, ordered, side="right") - 1
        # Runs of neighbouring rows of one piece: (first position in `ordered`, count).
        breaks = np.flatnonzero((np.diff(ordered) != 1) | (np.diff(pieces) != 0)) + 1
        firsts = np.concatenate([[0], breaks]).astype(np.int64)
        counts = np.diff(np.concatenate([firsts, [len(ordered)]])).astype(np.int64)
        if not len(ordered):
            return np.zeros((0, 0), dtype=np.int8)
        width = self._piece(int(pieces[0])).width
        out = np.empty((len(ordered), width), dtype=np.int8)

        def read(run: tuple[int, int]) -> None:
            at, count = run
            piece = self._piece(int(pieces[at]))
            local = int(ordered[at] - self.starts[pieces[at]])
            fd = os.open(piece.path, os.O_RDONLY)
            try:
                data = os.pread(fd, count * width, piece.offset + local * width)
            finally:
                os.close(fd)
            out[at : at + count] = np.frombuffer(data, dtype=np.int8).reshape(count, width)

        runs = list(zip(firsts.tolist(), counts.tolist(), strict=True))
        list(bounded_map(read, runs, workers=READERS))
        back = np.empty_like(out)
        back[order] = out
        return back

    def cosines(self, query: Floats, rows: Ints) -> Floats:
        """Exact cosines of a unit query with these crop rows (in that order)."""
        block = self.rows(rows).astype(np.float32)
        return ((block @ query) * self.scale[rows]).astype(np.float32)


@dataclass(frozen=True, slots=True)
class CropIndex:
    rule: CropRule
    mean: Floats
    components: Floats
    row_file: NDArray[np.int32]
    store: CropStore
    whole: WholeCrops

    def short_query(self, a3_full: Floats) -> Floats:
        """A unit MegaLoc query (8,448 numbers) on the crops' axes, renormalised."""
        coords = ((a3_full - self.mean) @ self.components.T).astype(np.float32)
        return coords / np.float32(np.linalg.norm(coords))

    def row_scores(self, a3_full: Floats, mask: NDArray[np.bool_]) -> Floats:
        """Each image row's best rescored crop's exact cosine; -inf for a row with no crop
        rescored, or filtered out."""
        picked = self.store.top_crops(self.short_query(a3_full), self.rule.rescore)
        sims = self.whole.cosines(a3_full, picked)
        files = len(self.whole.scale) // self.rule.per_file
        best = np.full(files, -np.inf, dtype=np.float32)
        np.maximum.at(best, picked // self.rule.per_file, sims)
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


def write_whole_index(directory: Path, names: Sequence[str], counts: Sequence[int]) -> None:
    starts: list[int] = []
    total = 0
    for count in counts:
        starts.append(total)
        total += count
    pieces = [{"piece": n, "start": s} for n, s in zip(names, starts, strict=True)]
    (directory / WHOLE_INDEX_FILE).write_text(json.dumps(pieces) + "\n")


def read_whole(directory: Path) -> WholeCrops:
    pieces = json.loads((directory / WHOLE_INDEX_FILE).read_text())
    return WholeCrops(
        folder=directory / WHOLE_DIR,
        starts=np.asarray([p["start"] for p in pieces], dtype=np.int64),
        names=tuple(p["piece"] for p in pieces),
        scale=np.load(directory / WHOLE_SCALE_FILE).astype(np.float32),
    )


def read_projection(directory: Path) -> tuple[Floats, Floats]:
    with np.load(directory / PROJECTION_FILE, allow_pickle=False) as data:
        return (
            np.asarray(data["mean"], dtype=np.float32),
            np.asarray(data["components"], dtype=np.float32),
        )


def read_numpy_crops(directory: Path, *, mmap: bool = False) -> NumpyCrops:
    coords = np.load(directory / COORDS_FILE, mmap_mode="r" if mmap else None)
    return NumpyCrops(coords=coords, scale=np.load(directory / SCALE_FILE).astype(np.float32))


def read_crops(directory: Path, store: CropStore | None = None) -> CropIndex | None:
    """The bundle's crops, picked by `store` (host memory and the CPU by default); None for
    a bundle without crops."""
    if not has_crops(directory):
        return None
    rule = read_rule(directory)
    mean, components = read_projection(directory)
    row_file = np.load(directory / ROW_FILE).astype(np.int32)
    whole = read_whole(directory)
    logger.info(
        "crops: %d axes to pick %d, %d image rows with crops, shortlist %d, weight %.2f",
        components.shape[0],
        rule.rescore,
        int((row_file >= 0).sum()),
        rule.shortlist,
        rule.weight,
    )
    return CropIndex(rule, mean, components, row_file, store or read_numpy_crops(directory), whole)
