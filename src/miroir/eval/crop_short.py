"""Shorter crop vectors on the crop slice (docs/search-levels.md, "Short crops").

Crops of all of POP kept whole would weigh 163 GB in int8: no machine here serves that. This
measure takes the 3x3 crops of the crop slice, kept whole by `miroir scw crop-short-slice`
(`miroir.jobs.crop_embed`), and scores them shortened every way the site could serve them,
with the slice test's own ranking and two measures (`miroir.eval.crops`):

- **Length.** Whole (8,448 numbers); the first 1,024, 512, 256 or 128 axes of the `pop-all`
  PCA (fitted on whole photos), or of a PCA fitted on the crops of the slice's other photos
  (never an answer's).
- **Scale.** A crop's score must sit on the scale of the whole photo's exact cosine, since an
  image takes the best of both. "norm": the cosine of the shortened vectors, as the site's
  whole-photo shortlist uses it; "approx": the whole cosine put back from the PCA
  (`miroir.jobs.crop_embed.ProjectedCrops`).
- **Storage.** float32, or int8 with one scale per crop (what a bundle would hold).
- **Shortlist.** At the site, only the crops of a few files would get a score: the files
  whose best crop ranks in the top 2,000 of 2.15 million (about 1 in 1,000). In the slice of
  19,787 files that is the top 20 (and 100 for a looser list). "Two-stage": the shortlist
  is picked with short vectors and scored with whole ones.
- **Weight.** An image's MegaLoc score is max(whole, weight * crop).
- **Precision.** MegaLoc in float32 (as the gallery) or bfloat16 (several times faster).

The table for each variant is the best crop score per query and file, which then replaces the
slice test's 3x3 table (`Scores.crops3`); everything else is the slice test's.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.eval.crops import (
    VARIANTS,
    QueryCase,
    Scores,
    by_group,
    look_alikes,
    rank_full,
    rank_slice,
    usable_rows,
)
from miroir.index.pca import fit_pca_rows
from miroir.web.store import quantize_rows

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]
Halves = NDArray[np.float16]

WHOLE, WITH_3X3 = VARIANTS[0], VARIANTS[3]
#: Rows scored at once (crops x queries stays a few hundred MB).
BLOCK = 16_384
#: Files whose crops the site would score, out of the slice's files (`SHORTLISTS`).
SHORTLISTS = (20, 100)
LENGTHS = (1024, 512, 256, 128)
#: Crop weights tried: an image scores max(whole, weight * crop).
WEIGHTS = (0.95, 0.9, 0.85, 0.8)


@dataclass(frozen=True, slots=True)
class SliceCrops:
    """Every crop vector the machine sent back, by precision, in file order."""

    files: tuple[str, ...]
    per_file: int
    vectors: Mapping[str, Halves]
    """Precision ("fp32", "bf16") -> (files x per_file, 8,448) unit vectors."""


def load_pieces(folder: Path) -> SliceCrops:
    paths = sorted((folder / "pieces").glob("*.npz"))
    if not paths:
        msg = f"{folder}: no pieces"
        raise FileNotFoundError(msg)
    files: list[str] = []
    parts: dict[str, list[Halves]] = {}
    per_file = 0
    failed = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            files += [str(f) for f in data["files"].tolist()]
            per_file = int(data["per_file"])
            failed += len(data["failed"])
            for key in data.files:
                if key.startswith("vectors_"):
                    parts.setdefault(key.removeprefix("vectors_"), []).append(data[key])
    vectors = {k: np.concatenate(v) for k, v in parts.items()}
    logger.info(
        "%d files (%d failed), %d crops each, %s", len(files), failed, per_file, list(vectors)
    )
    return SliceCrops(tuple(files), per_file, vectors)


# --- a form of the crops: how one crop scores against one query --------------------------------


@dataclass(frozen=True, slots=True)
class Form:
    """Crop scores for every crop and query: `score(rows)` gives (len(rows), queries)."""

    name: str
    score: Callable[[slice], Floats]
    rows: int


def whole_form(name: str, crops: Halves, queries: Floats) -> Form:
    def score(rows: slice) -> Floats:
        return (np.asarray(crops[rows], dtype=np.float32) @ queries.T).astype(np.float32)

    return Form(name, score, len(crops))


@dataclass(frozen=True, slots=True)
class Projection:
    mean: Floats
    components: Floats
    """(axes, 8,448), largest axis first."""


def projected(crops: Halves, projection: Projection, dims: int) -> tuple[Floats, Floats]:
    """(coords, mean_dot) of every crop on the first `dims` axes, in blocks."""
    mean = projection.mean
    axes = projection.components[:dims]
    coords = np.empty((len(crops), dims), dtype=np.float32)
    mean_dot = np.empty(len(crops), dtype=np.float32)
    for start in range(0, len(crops), BLOCK):
        x = np.asarray(crops[start : start + BLOCK], dtype=np.float32)
        coords[start : start + len(x)] = (x - mean) @ axes.T
        mean_dot[start : start + len(x)] = x @ mean
    return coords, mean_dot


def short_form(
    name: str,
    coords: Floats,
    mean_dot: Floats,
    queries: Floats,
    projection: Projection,
    *,
    scale: str,
    int8: bool,
) -> Form:
    """Crops on `coords.shape[1]` axes. `scale` "approx": the whole cosine put back
    (`ProjectedCrops`); "norm": the cosine of the shortened vectors."""
    dims = coords.shape[1]
    axes = projection.components[:dims]
    q_coords = (queries - projection.mean) @ axes.T
    q_offset = queries @ projection.mean - float(projection.mean @ projection.mean)
    stored = coords
    if int8:
        rows, steps = quantize_rows(coords)
        stored = rows.astype(np.float32) * steps[:, None]
    if scale == "norm":
        unit = stored / np.linalg.norm(stored, axis=1, keepdims=True)
        q_unit = q_coords / np.linalg.norm(q_coords, axis=1, keepdims=True)

        def score(rows: slice) -> Floats:
            return (unit[rows] @ q_unit.T).astype(np.float32)

    else:

        def score(rows: slice) -> Floats:
            sims = stored[rows] @ q_coords.T + mean_dot[rows, None] + q_offset[None, :]
            return sims.astype(np.float32)

    return Form(name, score, len(coords))


def best_per_file(form: Form, per_file: int) -> Floats:
    """(queries, files): each file's best crop score."""
    blocks: list[Floats] = []
    step = BLOCK - BLOCK % per_file
    for start in range(0, form.rows, step):
        sims = form.score(slice(start, min(start + step, form.rows)))
        blocks.append(sims.reshape(-1, per_file, sims.shape[1]).max(axis=1))
    return np.concatenate(blocks).T.astype(np.float32)


# --- tables into the slice test's scores ---------------------------------------------------------


def aligned(table: Floats, files: Sequence[str], columns: Sequence[str]) -> Halves:
    """`table` (queries, `files`) in the slice test's crop columns; -inf where a file has no
    crop here."""
    at = {f: i for i, f in enumerate(files)}
    out = np.full((table.shape[0], len(columns)), -np.inf, dtype=np.float32)
    for c, f in enumerate(columns):
        if f in at:
            out[:, c] = table[:, at[f]]
    return out.astype(np.float16)


def shortlisted(
    pick: Halves, value: Halves, size: int, hidden_cols: Sequence[NDArray[np.int64]]
) -> Halves:
    """`value` kept for each query's `size` best files by `pick` (its hidden files left
    out), -inf elsewhere."""
    out = np.full(value.shape, -np.inf, dtype=np.float16)
    for q in range(pick.shape[0]):
        row = pick[q].astype(np.float32)
        row[hidden_cols[q]] = -np.inf
        best = np.argpartition(-row, size)[:size]
        out[q, best] = value[q, best]
    return out


def measure(
    scores: Scores, table: Halves, cases: Sequence[QueryCase], *, weight: float = 1.0
) -> dict[str, object]:
    """The slice and full-index hit rates with this 3x3 table, and the look-alikes it brings."""
    crops = (table.astype(np.float32) * weight).astype(np.float16)
    mine = replace(scores, crops3={**scores.crops3, "g3": crops})
    rows = usable_rows(mine)
    ranked = rank_slice(mine, cases, [WHOLE, WITH_3X3], rows=rows)
    full = rank_full(mine, cases, [WHOLE, WITH_3X3])
    look: dict[str, dict[str, int]] = {}
    for set_id in sorted({c.set_id for c in cases}):
        idx = [i for i, c in enumerate(cases) if c.set_id == set_id]
        look[set_id] = look_alikes(
            [ranked.ranks[WHOLE.name][i] for i in idx],
            [ranked.ranks[WITH_3X3.name][i] for i in idx],
        )
    return {
        "slice": by_group(cases, {WITH_3X3.name: ranked.ranks[WITH_3X3.name]}, "set_id"),
        "full_index_estimate": by_group(cases, {WITH_3X3.name: full[WITH_3X3.name]}, "set_id"),
        "look_alikes": look,
    }


def baseline(scores: Scores, cases: Sequence[QueryCase]) -> dict[str, object]:
    rows = usable_rows(scores)
    ranked = rank_slice(scores, cases, [WHOLE], rows=rows)
    full = rank_full(scores, cases, [WHOLE])
    return {
        "slice": by_group(cases, ranked.ranks, "set_id"),
        "full_index_estimate": by_group(cases, full, "set_id"),
    }


def other_rows(crops: SliceCrops, answer_files: set[str]) -> NDArray[np.int64]:
    """Crop rows of files no query's answer holds: what a crop PCA may be fitted on."""
    keep = [i for i, f in enumerate(crops.files) if f not in answer_files]
    return np.asarray(
        [i * crops.per_file + k for i in keep for k in range(crops.per_file)], dtype=np.int64
    )


def fit_crop_pca(crops: Halves, rows: NDArray[np.int64], dims: int) -> Projection:
    pca = fit_pca_rows(crops[rows], dims, "crops of the slice's other photos")
    return Projection(pca.mean.astype(np.float32), pca.components.T.copy())


def storage_gb(crops: int, dims: int, *, int8: bool) -> float:
    """What `crops` crops weigh on `dims` axes (plus 8 bytes for the scale and mean dot)."""
    return round(crops * (dims * (1 if int8 else 2) + 8) / 1e9, 1)


# --- the whole measure -------------------------------------------------------------------------

#: Crops of all of POP at 3x3 (2.157 million stored files, 9 each), for the storage column.
POP_CROPS = 19_400_000
#: Crops the site would rescore whole, out of all of POP's, for `crop_two_stage`.
RESCORED = (5_000, 20_000, 50_000)


def per_crop(form: Form) -> Floats:
    """(crops, queries): every crop's score."""
    return np.concatenate(
        [form.score(slice(a, min(a + BLOCK, form.rows))) for a in range(0, form.rows, BLOCK)]
    )


def crop_two_stage(
    short: Floats, whole: Floats, size: int, per_file: int, hidden: Sequence[NDArray[np.int64]]
) -> Floats:
    """(queries, files): each query's `size` best crops by `short` (its hidden files' crops
    left out) scored by `whole`; a file none of whose crops is kept scores -inf."""
    files = short.shape[0] // per_file
    out = np.full((short.shape[1], files), -np.inf, dtype=np.float32)
    for q in range(short.shape[1]):
        col = short[:, q].copy()
        for f in hidden[q].tolist():
            col[f * per_file : (f + 1) * per_file] = -np.inf
        best = np.argpartition(-col, size)[:size]
        np.maximum.at(out[q], best // per_file, whole[best, q])
    return out


@dataclass(frozen=True, slots=True)
class SliceMeasure:
    """Every variant of the module doc over one slice (`run`)."""

    scores: Scores
    cases: Sequence[QueryCase]
    crops: SliceCrops
    queries: Floats
    columns: Sequence[str]
    pop_pca: Projection
    hidden: list[NDArray[np.int64]]
    tables: dict[str, Halves]
    variants: dict[str, dict[str, object]]

    def table(self, form: Form) -> Halves:
        if form.name not in self.tables:
            best = best_per_file(form, self.crops.per_file)
            self.tables[form.name] = aligned(best, self.crops.files, self.columns)
            logger.info("table %s", form.name)
        return self.tables[form.name]

    def add(self, name: str, table: Halves, weight: float = 1.0) -> None:
        self.variants[name] = measure(self.scores, table, self.cases, weight=weight)

    def whole_and_short(self, answer_files: set[str]) -> None:
        for precision, vectors in self.crops.vectors.items():
            name = f"whole-{precision}"
            self.add(name, self.table(whole_form(name, vectors, self.queries)))
        reference = self.crops.vectors["fp32"]
        coords, mean_dot = projected(reference, self.pop_pca, LENGTHS[0])
        for dims in LENGTHS:
            for scale in ("approx", "norm"):
                for int8 in (False, True):
                    name = f"pop-pca{dims}-{scale}" + ("-int8" if int8 else "")
                    form = short_form(
                        name, coords[:, :dims], mean_dot, self.queries, self.pop_pca,
                        scale=scale, int8=int8,
                    )  # fmt: skip
                    self.add(name, self.table(form))
        if "bf16" in self.crops.vectors:
            b_coords, b_dot = projected(self.crops.vectors["bf16"], self.pop_pca, 256)
            name = "pop-pca256-norm-int8-bf16"
            form = short_form(
                name, b_coords, b_dot, self.queries, self.pop_pca, scale="norm", int8=True
            )
            self.add(name, self.table(form))
        crop_pca = fit_crop_pca(reference, other_rows(self.crops, answer_files), 256)
        c_coords, c_dot = projected(reference, crop_pca, 256)
        for dims in (256, 128):
            name = f"crop-pca{dims}-norm-int8"
            form = short_form(
                name, c_coords[:, :dims], c_dot, self.queries, crop_pca, scale="norm", int8=True
            )
            self.add(name, self.table(form))

    def weights_and_shortlists(self) -> None:
        norms = [f"pop-pca{d}-norm-int8" for d in LENGTHS]
        for base in ("whole-fp32", *norms):
            for weight in WEIGHTS:
                self.add(f"{base} weight {weight}", self.tables[base], weight)
        whole = self.tables["whole-fp32"]
        for size in SHORTLISTS:
            self.add(f"whole-fp32 top {size}", shortlisted(whole, whole, size, self.hidden))
            for pick in norms[1:]:
                kept = shortlisted(self.tables[pick], self.tables[pick], size, self.hidden)
                for weight in (1.0, *WEIGHTS):
                    self.add(f"{pick} weight {weight} top {size}", kept, weight)
                kept = shortlisted(self.tables[pick], whole, size, self.hidden)
                self.add(f"two-stage {pick} -> whole, top {size}", kept)

    def crop_level_two_stage(self) -> None:
        """Short coordinates pick crops (not files), whole vectors rescore them: what the
        site can afford to read from disk, `RESCORED` crops of all of POP per search."""
        reference = self.crops.vectors["fp32"]
        whole = per_crop(whole_form("whole", reference, self.queries))
        coords, mean_dot = projected(reference, self.pop_pca, 256)
        for dims in (256, 128):
            form = short_form(
                f"pca{dims}", coords[:, :dims], mean_dot, self.queries, self.pop_pca,
                scale="norm", int8=True,
            )  # fmt: skip
            short = per_crop(form)
            for rescored in RESCORED:
                size = max(1, round(rescored * len(reference) / POP_CROPS))
                table = crop_two_stage(short, whole, size, self.crops.per_file, self.hidden)
                kept = aligned(table, self.crops.files, self.columns)
                self.add(f"crop two-stage pca{dims}, {rescored:,} of POP's crops ({size})", kept)

    def shift(self, gallery_short: NDArray[np.float16]) -> None:
        rows = column_rows(self.scores, len(self.columns))
        exact = self.scores.whole3[:, rows].astype(np.float32)
        for dims in LENGTHS[1:]:
            short = whole_short(self.queries, gallery_short[rows], self.pop_pca, dims)
            moved = shifted(self.tables[f"pop-pca{dims}-norm-int8"], short, exact)
            self.add(f"shift pca{dims}", moved)


def run_measure(
    scores: Scores,
    cases: Sequence[QueryCase],
    crops: SliceCrops,
    queries: Floats,
    *,
    columns: Sequence[str],
    pop_pca: Projection,
    answer_files: set[str],
    gallery_short: NDArray[np.float16] | None = None,
) -> dict[str, object]:
    """Every variant of the module doc. `queries`: unit MegaLoc vectors in `scores.query_ids`
    order; `columns`: the slice test's crop columns."""
    by_id = {c.query_id: c for c in cases}
    hidden: list[NDArray[np.int64]] = []
    for query_id in scores.query_ids:
        case = by_id.get(query_id)
        cols = scores.row_file[case.hidden_rows] if case is not None else np.zeros(0, np.int64)
        hidden.append(cols[cols >= 0])
    run = SliceMeasure(scores, cases, crops, queries, columns, pop_pca, hidden, {}, {})
    out: dict[str, object] = {"baseline": baseline(scores, cases)}
    run.whole_and_short(answer_files)
    old = scores.crops3["g3"].astype(np.float32)
    new = run.tables["whole-fp32"].astype(np.float32)
    both = np.isfinite(old) & np.isfinite(new)
    out["check_against_slice_test"] = {
        "max_abs_diff": float(np.abs(old[both] - new[both]).max()),
        "cells": int(both.sum()),
    }
    run.weights_and_shortlists()
    run.crop_level_two_stage()
    if gallery_short is not None:
        run.shift(gallery_short)
    out["variants"] = run.variants
    out["storage_gb_all_pop"] = {
        "whole int8": storage_gb(POP_CROPS, 8448, int8=True),
        **{f"pca{d} int8": storage_gb(POP_CROPS, d, int8=True) for d in LENGTHS},
        **{f"pca{d} float16": storage_gb(POP_CROPS, d, int8=False) for d in LENGTHS},
    }
    out["crops"] = {"files": len(crops.files), "per_file": crops.per_file}
    return out


# --- crops judged against their own photo in the short space ------------------------------------


def column_rows(scores: Scores, columns: int) -> NDArray[np.int64]:
    """One gallery row of each crop column's file (rows sharing a file share its vectors)."""
    out = np.full(columns, -1, dtype=np.int64)
    for row, col in enumerate(scores.row_file.tolist()):
        if col >= 0 and out[col] < 0:
            out[col] = row
    return out


def whole_short(
    queries: Floats, gallery: NDArray[np.floating], projection: Projection, dims: int
) -> Floats:
    """(queries, rows): the cosine of the whole photos on the first `dims` PCA axes, as the
    site's shortlist holds them (`a3.npy`, unit rows on 1,024 axes, cut and renormalised)."""
    g = np.asarray(gallery[:, :dims], dtype=np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    q = (queries - projection.mean) @ projection.components[:dims].T
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return (q @ g.T).astype(np.float32)


def shifted(crop: Halves, short_whole: Floats, exact_whole: Floats) -> Halves:
    """A crop's short cosine moved by its own photo's gap between the exact and the short
    whole cosine: crop_short - whole_short + whole_exact. The short space judges whether the
    crop matches better than its photo; the exact cosine gives the scale."""
    moved = crop.astype(np.float32) - short_whole + exact_whole
    return moved.astype(np.float16)
