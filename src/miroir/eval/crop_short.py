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


def run_measure(
    scores: Scores,
    cases: Sequence[QueryCase],
    crops: SliceCrops,
    queries: Floats,
    *,
    columns: Sequence[str],
    pop_pca: Projection,
    answer_files: set[str],
) -> dict[str, object]:
    """Every variant of the module doc. `queries`: unit MegaLoc vectors in `scores.query_ids`
    order; `columns`: the slice test's crop columns."""
    per_file = crops.per_file
    hidden: list[NDArray[np.int64]] = []
    by_id = {c.query_id: c for c in cases}
    for query_id in scores.query_ids:
        case = by_id.get(query_id)
        cols = scores.row_file[case.hidden_rows] if case is not None else np.zeros(0, np.int64)
        hidden.append(cols[cols >= 0])

    tables: dict[str, Halves] = {}

    def table(form: Form) -> Halves:
        if form.name not in tables:
            tables[form.name] = aligned(best_per_file(form, per_file), crops.files, columns)
            logger.info("table %s", form.name)
        return tables[form.name]

    out: dict[str, object] = {"baseline": baseline(scores, cases)}
    variants: dict[str, dict[str, object]] = {}
    for precision, vectors in crops.vectors.items():
        variants[f"whole-{precision}"] = measure(
            scores, table(whole_form(f"whole-{precision}", vectors, queries)), cases
        )
    reference = crops.vectors["fp32"]
    old = scores.crops3["g3"].astype(np.float32)
    new = tables["whole-fp32"].astype(np.float32)
    both = np.isfinite(old) & np.isfinite(new)
    out["check_against_slice_test"] = {
        "max_abs_diff": float(np.abs(old[both] - new[both]).max()),
        "cells": int(both.sum()),
    }

    coords, mean_dot = projected(reference, pop_pca, LENGTHS[0])
    for dims in LENGTHS:
        for scale in ("approx", "norm"):
            name = f"pop-pca{dims}-{scale}"
            form = short_form(
                name, coords[:, :dims], mean_dot, queries, pop_pca, scale=scale, int8=False
            )
            variants[name] = measure(scores, table(form), cases)
    for dims in (256, 128):
        name = f"pop-pca{dims}-approx-int8"
        form = short_form(
            name, coords[:, :dims], mean_dot, queries, pop_pca, scale="approx", int8=True
        )
        variants[name] = measure(scores, table(form), cases)
    if "bf16" in crops.vectors:
        b_coords, b_mean_dot = projected(crops.vectors["bf16"], pop_pca, 256)
        name = "pop-pca256-approx-int8-bf16"
        form = short_form(name, b_coords, b_mean_dot, queries, pop_pca, scale="approx", int8=True)
        variants[name] = measure(scores, table(form), cases)
        del b_coords
    del coords

    crop_pca = fit_crop_pca(reference, other_rows(crops, answer_files), 256)
    c_coords, c_mean_dot = projected(reference, crop_pca, 256)
    for dims in (256, 128):
        name = f"crop-pca{dims}-approx-int8"
        form = short_form(
            name, c_coords[:, :dims], c_mean_dot, queries, crop_pca, scale="approx", int8=True
        )
        variants[name] = measure(scores, table(form), cases)
    del c_coords

    for base in ("whole-fp32", "pop-pca256-approx-int8", "pop-pca128-approx-int8"):
        for weight in (0.95, 0.9):
            variants[f"{base} weight {weight}"] = measure(
                scores, tables[base], cases, weight=weight
            )

    for size in SHORTLISTS:
        for base in ("whole-fp32", "pop-pca256-approx-int8", "pop-pca128-approx-int8"):
            kept = shortlisted(tables[base], tables[base], size, hidden)
            variants[f"{base} top {size}"] = measure(scores, kept, cases)
        for pick in ("pop-pca256-approx-int8", "pop-pca128-approx-int8"):
            kept = shortlisted(tables[pick], tables["whole-fp32"], size, hidden)
            variants[f"two-stage {pick} -> whole, top {size}"] = measure(scores, kept, cases)

    out["variants"] = variants
    out["storage_gb_all_pop"] = {
        "whole int8": storage_gb(POP_CROPS, 8448, int8=True),
        **{f"pca{d} int8": storage_gb(POP_CROPS, d, int8=True) for d in LENGTHS},
        **{f"pca{d} float16": storage_gb(POP_CROPS, d, int8=False) for d in LENGTHS},
    }
    out["crops"] = {"files": len(crops.files), "per_file": per_file}
    return out
