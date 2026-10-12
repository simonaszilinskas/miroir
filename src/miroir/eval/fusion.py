"""Combining several image models' scores into one ranking (experiments.md step G).

Both methods work on entity scores (`scoring.EntityScores`), so every model ranks the same
buildings and objects before they are combined:

- G1, weighted sum: each model's scores are min-max normalised per query (best entity 1, worst
  0), then summed with weights. This is what OpenSearch's hybrid pipeline does.
- G2, reciprocal-rank fusion: each entity gets the weighted sum of 1 / (RRF_K + rank) over
  models. Only ranks count, so a model's score scale does not matter.

- G4, routed weighted sum: G1 with weights set per query (`fuse_per_query`, `router`).

An entity hidden from a query (score -inf) stays hidden after fusion.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np

from miroir.eval.scoring import EntityScores, Floats

#: The usual constant of reciprocal-rank fusion (Cormack et al., 2009).
RRF_K = 60


class Method(StrEnum):
    WEIGHTED_SUM = "G1"
    RECIPROCAL_RANK = "G2"


def fuse(parts: Sequence[EntityScores], weights: Sequence[float], method: Method) -> EntityScores:
    """One set of entity scores from several models' scores on the same queries and gallery."""
    first = _check(parts, weights)
    if any(w <= 0 for w in weights):
        msg = f"weights must be positive: {weights}"
        raise ValueError(msg)
    rows = np.broadcast_to(
        np.asarray(weights, dtype=np.float32), (len(first.query_ids), len(parts))
    )
    return _combine(parts, rows, method)


def fuse_per_query(parts: Sequence[EntityScores], weights: Floats) -> EntityScores:
    """G1 with each query's own weights: `weights[i, m]` is model m's weight for query i.

    A weight may be 0, leaving that model out for that query (G4 routes this way).
    """
    first = _check(parts, [1.0] * len(parts))
    shape = (len(first.query_ids), len(parts))
    if weights.shape != shape or (weights < 0).any():
        msg = f"per-query weights need shape {shape} and no negative value, got {weights.shape}"
        raise ValueError(msg)
    return _combine(parts, weights, Method.WEIGHTED_SUM)


def _combine(parts: Sequence[EntityScores], weights: Floats, method: Method) -> EntityScores:
    first = parts[0]
    hidden = np.zeros(first.values.shape, dtype=bool)
    total: Floats = np.zeros(first.values.shape, dtype=np.float32)
    for m, part in enumerate(parts):
        hidden |= ~np.isfinite(part.values)
        column = weights[:, m : m + 1].astype(np.float32)
        if method is Method.WEIGHTED_SUM:
            total += column * min_max(part.values)
        else:
            total += column / (RRF_K + ranks(part.values))
    total[hidden] = -np.inf
    return EntityScores(
        query_ids=first.query_ids,
        entity_ids=first.entity_ids,
        values=total,
        gallery_images=first.gallery_images,
    )


def min_max(values: Floats) -> Floats:
    """Each row scaled to [0, 1] over its finite values. Hidden entities (-inf) become 0."""
    finite = np.isfinite(values)
    low = np.where(finite, values, np.inf).min(axis=1, keepdims=True)
    high = np.where(finite, values, -np.inf).max(axis=1, keepdims=True)
    span = np.where(high > low, high - low, 1.0)
    return np.where(finite, (values - low) / span, 0.0).astype(np.float32)


def ranks(values: Floats) -> Floats:
    """Rank of each entity in its row, from 1. Ties go to the lower entity id, as in scoring."""
    order = np.argsort(-values, axis=1, kind="stable")
    out: Floats = np.empty(values.shape, dtype=np.float32)
    positions = np.arange(1, values.shape[1] + 1, dtype=np.float32)
    np.put_along_axis(out, order, np.broadcast_to(positions, values.shape), axis=1)
    return out


def split_weight(first: float, count: int) -> tuple[float, ...]:
    """Weights for `count` models: `first` for the first, the rest shared equally by the others."""
    if count < 2 or not 0 < first < 1:
        msg = f"need 2+ models and a first weight in (0, 1), got {count} and {first}"
        raise ValueError(msg)
    rest = (1 - first) / (count - 1)
    return (first, *([rest] * (count - 1)))


def _check(parts: Sequence[EntityScores], weights: Sequence[float]) -> EntityScores:
    if len(parts) < 2 or len(parts) != len(weights):
        msg = f"fusion needs 2+ models with one weight each: {len(parts)} and {len(weights)}"
        raise ValueError(msg)
    first = parts[0]
    for part in parts[1:]:
        if part.query_ids != first.query_ids or part.entity_ids != first.entity_ids:
            msg = "models scored different queries or galleries"
            raise ValueError(msg)
    return first
