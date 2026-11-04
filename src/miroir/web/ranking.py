"""Scores per image in, ranked entities out (functional spec §6). NumPy only.

The same steps as the evaluation (`miroir.eval.scoring`, `fusion`, `router`), so the site
ranks as the measured stack does:

1. each signal (a model's cosines, BM25) is a score per image; images a filter removes
   score -inf;
2. an entity scores the best of its images, per signal;
3. each signal is min-max scaled over the entities left (best 1, worst 0);
4. signals are added with weights: G4's routed weights for a photo, the text weights for
   words, words counting `refine` when they refine a photo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from miroir.web.catalog import Catalog

Floats = NDArray[np.float32]
Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]


def entity_max(values: Floats, catalog: Catalog) -> Floats:
    """Best image score per entity; -inf for an entity with no image left (or none at all)."""
    if values.shape != (len(catalog.images),):
        msg = f"one score per image expected, got shape {values.shape}"
        raise ValueError(msg)
    out = np.full(catalog.entity_count, -np.inf, dtype=np.float32)
    sorted_values = values[catalog.entity_order].astype(np.float32)
    out[catalog.entities_with_images] = np.maximum.reduceat(sorted_values, catalog.entity_starts)
    return out


def min_max(values: Floats) -> Floats:
    """Scaled to [0, 1] over finite values; -inf (removed) stays -inf. Flat rows become 0."""
    finite = np.isfinite(values)
    out = np.full(values.shape, -np.inf, dtype=np.float32)
    if not finite.any():
        return out
    low, high = float(values[finite].min()), float(values[finite].max())
    span = high - low if high > low else 1.0
    out[finite] = (values[finite] - low) / span
    return out


@dataclass(frozen=True, slots=True)
class Signal:
    """One score per entity and its weight in the sum."""

    per_entity: Floats
    weight: float


def fuse(signals: Sequence[Signal]) -> Floats:
    """Weighted sum of min-max scaled signals. An entity removed from any signal stays out."""
    if not signals:
        msg = "nothing to fuse"
        raise ValueError(msg)
    total = np.zeros(signals[0].per_entity.shape, dtype=np.float32)
    for signal in signals:
        total += np.float32(signal.weight) * np.nan_to_num(min_max(signal.per_entity), neginf=0.0)
        total[~np.isfinite(signal.per_entity)] = -np.inf
    return total


def weighted(a7: Floats, a3: Floats, a7_weight: float) -> Floats:
    """`a7_weight * a7 + (1 - a7_weight) * a3`, leaving out a term weighted 0 (so a MegaLoc
    score of -inf, off the shortlist, cannot turn an EmbeddingGemma 2-only score into NaN)."""
    if a7_weight >= 1:
        return np.asarray(a7, dtype=np.float32)
    if a7_weight <= 0:
        return np.asarray(a3, dtype=np.float32)
    return (a7_weight * a7 + (1 - a7_weight) * a3).astype(np.float32)


@dataclass(frozen=True, slots=True)
class PhotoRanking:
    """A photo's ranking: both models' best cosine per entity, weighted by the photo's
    routing bucket (G5) and fused. Every photo search and every measurement of one ranks by
    this (`rank_photo`), so what is measured is what the site shows."""

    e7: Floats
    e3: Floats
    a7_weight: float
    signals: tuple[Signal, ...]
    fused: Floats

    def confidence(self, entity: int) -> float:
        """The routed mix of the entity's raw cosines: what "not found" and the card levels
        read (T8 measures it)."""
        mixed = weighted(self.e7[entity : entity + 1], self.e3[entity : entity + 1], self.a7_weight)
        return float(mixed[0])


def rank_photo(e7: Floats, e3: Floats, a7_weight: float) -> PhotoRanking:
    """Both models' entity scores in, the photo's fused ranking out (a model weighted 0 is
    left out, so its -inf cannot remove an entity)."""
    signals = tuple(s for s in (Signal(e7, a7_weight), Signal(e3, 1 - a7_weight)) if s.weight > 0)
    return PhotoRanking(e7=e7, e3=e3, a7_weight=a7_weight, signals=signals, fused=fuse(signals))


def top_entities(scores: Floats, k: int) -> Ints:
    """Rows of the `k` best entities, best first; ties go to the lower row (entity id order)."""
    alive = np.flatnonzero(np.isfinite(scores))
    order = np.lexsort((alive, -scores[alive]))
    return alive[order[:k]].astype(np.int64)


def best_image(rows: Ints, per_image: Floats, mask: Bools) -> int | None:
    """The image row, among an entity's `rows`, that the query matches best (filters kept)."""
    allowed = rows[mask[rows]]
    if allowed.size == 0:
        return None
    return int(allowed[int(np.argmax(per_image[allowed]))])
