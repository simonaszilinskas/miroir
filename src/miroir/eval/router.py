"""The building-or-object guess, and fusion weights routed by it (experiments.md G4).

The guess is a logistic regression on vectors that already exist (no model runs here). It
learns from gallery images whose record says what they show (`photo_type`), leaving out every
building or object any test set uses, tune and sealed (`sets.entities_in_sets`). The website
reads it from its bundle (`guess.npz`, `miroir index guess`).

A `Route` sets the first model's weight from p(building), as a blend

    weight = p(building) * w_building + (1 - p(building)) * w_object

or as a hard switch at `SWITCH_AT`; the other models share the rest. `rank_routed` ranks a set
that way, fast enough for a grid (`miroir.eval.router3`). The G4 weight search itself is
retired (docs/experiments.md, "Retired code").
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from miroir.config import SEED, Settings
from miroir.eval.fusion import fuse_per_query
from miroir.eval.scoring import (
    EntityScores,
    Floats,
    ScoreResult,
    SearchQuery,
    rank_entities,
)
from miroir.eval.sets import entities_in_sets, load_queries
from miroir.eval.vectors import VectorSet
from miroir.records.model import Collection, Record
from miroir.records.store import read_records

logger = logging.getLogger(__name__)

Float64s = NDArray[np.float64]

#: First-model weights tried for buildings and for objects. 1.0 and 0.0 leave a model out.
ROUTE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
#: Sets whose tune queries the guess is checked on (it leaves out the entities of every set).
QUERY_SETS = ("T1", "T3", "T5", "T8")
#: L2 penalties tried for the logistic regression, on a held-out share of the gallery.
PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
HELD_OUT_SHARE = 0.2
SWITCH_AT = 0.5


class PhotoType(StrEnum):
    BUILDING = "building"
    OBJECT = "object"


def photo_type(record: Record) -> PhotoType | None:
    """What a record's photos show, from the record alone. None when it does not say.

    Joconde and Rose Valland hold objects and artworks. A Mémoire photo shows a Palissy object
    when its entity is a PM id, a Mérimée building when it is a PA id; a photo linked to
    neither (entity = its own ref) is left out.
    """
    if record.collection in (Collection.JOCONDE, Collection.ROSE_VALLAND):
        return PhotoType.OBJECT
    if record.entity_id.startswith("PM"):
        return PhotoType.OBJECT
    if record.entity_id.startswith("PA"):
        return PhotoType.BUILDING
    return None


# --- the guess -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypeGuess:
    """A fitted logistic regression giving p(building) from an image's vectors."""

    runs: tuple[str, ...]
    """Vector runs whose rows, each L2-normalised, are joined into one feature row."""
    mean: Float64s
    scale: Float64s
    coef: Float64s
    bias: float
    penalty: float

    def p_building(self, features: NDArray[np.floating]) -> Float64s:
        z = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return _sigmoid(z @ self.coef + self.bias)


@dataclass(frozen=True, slots=True)
class GuessReport:
    guess: TypeGuess
    trained_on: int
    """Gallery images the final fit learned from."""
    left_out_entities: int
    """Buildings and objects of tune queries kept out of training."""
    held_out_accuracy: float
    """Balanced accuracy on a held-out share of the gallery, by building or object."""
    query_accuracy: tuple[tuple[str, int, float], ...]
    """Per set: queries whose true type is known, and the share guessed right."""


def fit_logistic(
    x: NDArray[np.floating], y: Float64s, sample_weight: Float64s, penalty: float
) -> tuple[Float64s, float]:
    """L2-penalised logistic regression by Newton steps with conjugate gradients.

    Minimises the weighted mean log-loss plus `penalty / 2 * |coef|^2` (the bias is free).
    Returns (coef, bias). `x` should be standardised; it is never copied.
    """
    n, d = x.shape
    if y.shape != (n,) or sample_weight.shape != (n,) or penalty <= 0:
        msg = f"fit_logistic: {x.shape} rows, {y.shape} labels, penalty {penalty}"
        raise ValueError(msg)
    s = sample_weight / sample_weight.sum()
    theta = np.zeros(d + 1)

    def margins(t: Float64s) -> Float64s:
        return np.asarray(x @ t[:d], dtype=np.float64) + t[d]

    def loss(t: Float64s) -> float:
        m = margins(t)
        # log(1 + e^m) - y m, written to stay finite for large |m|.
        per = np.logaddexp(0.0, m) - y * m
        return float(s @ per + 0.5 * penalty * t[:d] @ t[:d])

    for _ in range(50):
        p = _sigmoid(margins(theta))
        r = s * (p - y)
        grad = np.append(np.asarray(x.T @ r, dtype=np.float64) + penalty * theta[:d], r.sum())
        if np.linalg.norm(grad) < 1e-7:
            break
        h = s * p * (1 - p)

        def hess(v: Float64s, h: Float64s = h) -> Float64s:
            u = h * (np.asarray(x @ v[:d], dtype=np.float64) + v[d])
            return np.append(np.asarray(x.T @ u, dtype=np.float64) + penalty * v[:d], u.sum())

        step = _conjugate_gradient(hess, -grad)
        before, rate = loss(theta), 1.0
        while loss(theta + rate * step) > before + 1e-4 * rate * (grad @ step) and rate > 1e-6:
            rate /= 2
        theta = theta + rate * step
    return theta[:d], float(theta[d])


def train_guess(
    runs: Sequence[VectorSet],
    labels: Mapping[str, PhotoType],
    entity_of: Mapping[str, str],
    *,
    seed: int = SEED,
) -> tuple[TypeGuess, float]:
    """Fit the guess on labelled images; pick the penalty on a held-out share of entities.

    Returns the guess refitted on every labelled image, and its balanced accuracy on the
    held-out share when fitted without it.
    """
    image_ids = sorted(labels)
    x = features(runs, image_ids)
    y = np.asarray([labels[i] is PhotoType.BUILDING for i in image_ids], dtype=np.float64)
    entities = sorted({entity_of[i] for i in image_ids})
    random.Random(seed).shuffle(entities)
    held = set(entities[: round(HELD_OUT_SHARE * len(entities))])
    test = np.asarray([entity_of[i] in held for i in image_ids], dtype=bool)
    tried: dict[float, float] = {}
    for penalty in PENALTIES:
        guess = _fit(runs, x[~test], y[~test], penalty)
        tried[penalty] = balanced_accuracy(y[test], guess.p_building(x[test]))
        logger.info("penalty %g: held-out balanced accuracy %.3f", penalty, tried[penalty])
    # Best held-out accuracy; a tie goes to the larger penalty, the simpler fit.
    best = max(sorted(PENALTIES, reverse=True), key=lambda p: tried[p])
    return _fit(runs, x, y, best), tried[best]


def balanced_accuracy(y: Float64s, p: Float64s) -> float:
    """Mean of the share right among buildings and the share right among objects."""
    guessed = p > SWITCH_AT
    truth = y > SWITCH_AT
    shares = [float((guessed[truth == side] == side).mean()) for side in (True, False)]
    return float(np.mean(shares))


def features(runs: Sequence[VectorSet], image_ids: Sequence[str]) -> NDArray[np.float32]:
    """Each image's vectors from every run, side by side."""
    blocks: list[NDArray[np.float32]] = []
    for run in runs:
        row_of = run.index()
        missing = [i for i in image_ids if i not in row_of]
        if missing:
            msg = f"{run.info.option}: {len(missing)} image(s) without a vector, e.g. {missing[:3]}"
            raise ValueError(msg)
        rows = np.fromiter((row_of[i] for i in image_ids), dtype=np.int64, count=len(image_ids))
        blocks.append(np.asarray(run.vectors[rows], dtype=np.float32))
    return np.concatenate(blocks, axis=1)


def _fit(
    runs: Sequence[VectorSet], x: NDArray[np.float32], y: Float64s, penalty: float
) -> TypeGuess:
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64) + 1e-6
    z = ((x - mean) / scale).astype(np.float32)
    # Each class weighs the same in total, so the many building photos do not swamp objects.
    buildings = y.sum()
    weight = np.where(y > SWITCH_AT, 0.5 / buildings, 0.5 / (y.size - buildings))
    coef, bias = fit_logistic(z, y, weight, penalty)
    return TypeGuess(
        runs=tuple(r.info.option for r in runs),
        mean=mean,
        scale=scale,
        coef=coef,
        bias=bias,
        penalty=penalty,
    )


def _conjugate_gradient(
    apply: Callable[[Float64s], Float64s], b: Float64s, *, steps: int = 100, tol: float = 1e-6
) -> Float64s:
    out = np.zeros_like(b)
    r = b.copy()
    d = r.copy()
    rr = r @ r
    for _ in range(steps):
        if np.sqrt(rr) <= tol * np.linalg.norm(b):
            break
        ad = apply(d)
        alpha = rr / (d @ ad)
        out += alpha * d
        r -= alpha * ad
        new = r @ r
        d = r + (new / rr) * d
        rr = new
    return out


def _sigmoid(m: Float64s) -> Float64s:
    return np.exp(-np.logaddexp(0.0, -m))


# --- routing ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Route:
    """First-model weights for building and object photos; hard: switch on p > 0.5."""

    building: float
    object: float
    hard: bool = False

    def first_weights(self, p_building: Float64s) -> Float64s:
        if self.hard:
            return np.where(p_building > SWITCH_AT, self.building, self.object)
        return p_building * self.building + (1 - p_building) * self.object


@dataclass(frozen=True, slots=True)
class RoutedSet:
    """One set's queries, each model's entity scores, and each query's p(building)."""

    queries: tuple[SearchQuery, ...]
    parts: tuple[EntityScores, ...]
    scaled: tuple[Floats, ...]
    """Each model's scores, min-max normalised per query, computed once for the grid."""
    p_building: Float64s


def model_weights(first: Float64s, count: int) -> Floats:
    """Per-query weights: `first` for model 0, the rest shared by the others."""
    rest = (1 - first) / (count - 1)
    return np.column_stack([first, *([rest] * (count - 1))]).astype(np.float32)


def routed(routed_set: RoutedSet, route: Route) -> EntityScores:
    weights = model_weights(route.first_weights(routed_set.p_building), len(routed_set.parts))
    return fuse_per_query(routed_set.parts, weights)


def rank_routed(routed_set: RoutedSet, route: Route) -> ScoreResult:
    """Fast path of `rank_entities(routed(...))` for the grid: reuses the min-max scores."""
    first = routed_set.parts[0]
    weights = model_weights(route.first_weights(routed_set.p_building), len(routed_set.parts))
    total = np.zeros(first.values.shape, dtype=np.float32)
    hidden = np.zeros(first.values.shape, dtype=bool)
    for m, (part, scaled) in enumerate(zip(routed_set.parts, routed_set.scaled, strict=True)):
        total += weights[:, m : m + 1] * scaled
        hidden |= ~np.isfinite(part.values)
    total[hidden] = -np.inf
    fused = EntityScores(first.query_ids, first.entity_ids, total, first.gallery_images)
    return rank_entities(fused, routed_set.queries)


# --- the whole run ---------------------------------------------------------------------------


def photo_types(settings: Settings) -> dict[str, PhotoType]:
    """Each image's photo type where its record says it (`photo_type`)."""
    records = read_records(settings.records_parquet, settings.images_parquet)
    out: dict[str, PhotoType] = {}
    unknown = 0
    for record in records:
        kind = photo_type(record)
        if kind is None:
            unknown += len(record.images)
            continue
        out.update((i.image_id, kind) for i in record.images)
    logger.info("photo type known for %d images, unknown for %d", len(out), unknown)
    return out


def guess_report(
    settings: Settings,
    guessing: Sequence[VectorSet],
    gallery_ids: Sequence[str],
    entity_of: Mapping[str, str],
    type_of_image: Mapping[str, PhotoType],
    *,
    query_sets: Sequence[str] = QUERY_SETS,
) -> GuessReport:
    """Train the guess on gallery images, every entity of every test set left out (both
    halves, `entities_in_sets`); report how right it is on the tune queries of `query_sets`."""
    queries = {s: load_queries(settings, s, "tune") for s in query_sets}
    left_out = entities_in_sets(settings)
    labels = training_labels(gallery_ids, entity_of, type_of_image, left_out)
    skipped = sum(1 for i in gallery_ids if i not in type_of_image)
    logger.info(
        "guess trains on %d gallery images (%d without a known type, %d entities left out)",
        len(labels),
        skipped,
        len(left_out),
    )
    guess, held_out = train_guess(guessing, labels, entity_of)
    # Queries whose image every guess run embedded (closed-model runs hold publishable ones).
    embedded = set(guessing[0].ids).intersection(*(r.ids for r in guessing[1:]))
    accuracy: list[tuple[str, int, float]] = []
    for set_id, qs in queries.items():
        known = [
            q for q in qs if q.query_image_id in type_of_image and q.query_image_id in embedded
        ]
        x = features(guessing, [q.query_image_id for q in known])
        truth = np.asarray([type_of_image[q.query_image_id] is PhotoType.BUILDING for q in known])
        right = (guess.p_building(x) > SWITCH_AT) == truth
        accuracy.append((set_id, len(known), float(right.mean())))
    return GuessReport(
        guess=guess,
        trained_on=len(labels),
        left_out_entities=len(left_out),
        held_out_accuracy=held_out,
        query_accuracy=tuple(accuracy),
    )


def training_labels(
    gallery_ids: Sequence[str],
    entity_of: Mapping[str, str],
    type_of_image: Mapping[str, PhotoType],
    left_out: frozenset[str],
) -> dict[str, PhotoType]:
    """The gallery images the guess learns from: those whose record gives the type, minus
    every image of an entity a test set uses (its query photos are gallery images too)."""
    return {
        i: type_of_image[i]
        for i in gallery_ids
        if i in type_of_image and entity_of[i] not in left_out
    }
