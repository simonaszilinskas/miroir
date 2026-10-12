"""G4: fusion weights chosen per photo by a building-or-object guess (experiments.md step G).

One G1 weight cannot suit every photo: T1 (archive photos of buildings) wants EmbeddingGemma 2,
T5 (museum objects) wants MegaLoc. G4 guesses from the query's own vectors whether it shows a
building or an object, and sets the first model's weight from that guess:

    weight = p(building) * w_building + (1 - p(building)) * w_object

or, as a hard switch, `w_building` when p(building) > 0.5 and `w_object` otherwise. The other
models share the rest equally, as in G1.

The guess is a logistic regression on vectors that already exist (no model runs here). It
learns from gallery images whose record says what they show (`photo_type`), leaving out every
building or object that appears in a T1, T3, T5 or T8 tune query. The two weights come from a
small grid, one pair for all scored sets together (mean hit@10 over sets), never one per set.

With only one building set (T1), `w_building` is in effect tuned on T1 alone. A two-fold check
(`cross_fitted`) picks the weights on one half of each set's buildings and objects and scores
the other half, so the gain over G1 can be read without that optimism.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from miroir.config import SEED, Settings, eval_result_path
from miroir.eval.fusion import fuse_per_query, min_max
from miroir.eval.notfound import Confidence, NotFoundScore, confidences, not_found_score
from miroir.eval.results import RunResult, set_score, write_result
from miroir.eval.run import publishable_only, search_queries
from miroir.eval.scoring import (
    EntityScores,
    Floats,
    ScoreResult,
    SearchQuery,
    entity_scores,
    per_query_values,
    rank_entities,
)
from miroir.eval.sets import Query, load_queries, read_gallery
from miroir.eval.stack import (
    IN_INDEX_SETS,
    NOT_FOUND_SET,
    Model,
    combined_info,
    load_model,
    not_found_queries,
)
from miroir.eval.vectors import VectorSet, read_vectors
from miroir.records.model import Collection, Record
from miroir.records.store import read_images, read_records

logger = logging.getLogger(__name__)

Float64s = NDArray[np.float64]

#: First-model weights tried for buildings and for objects. 1.0 and 0.0 leave a model out.
ROUTE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
#: Sets whose tune queries' buildings and objects never train the guess.
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

    def subset(self, keep: NDArray[np.bool_]) -> RoutedSet:
        rows = np.flatnonzero(keep)
        return RoutedSet(
            queries=tuple(self.queries[i] for i in rows),
            parts=tuple(_rows(p, rows) for p in self.parts),
            scaled=tuple(s[rows] for s in self.scaled),
            p_building=self.p_building[rows],
        )


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


def pick_route(
    sets: Sequence[RoutedSet], *, hard: bool, tied: bool = False
) -> tuple[Route, tuple[float, float]]:
    """The grid pair with the best mean hit@10 over sets (then mAP), one pair for all sets.

    `tied` forces the two weights equal: plain G1 on the same grid. A tie in score goes to
    the pair with the weights closest together, then closest to equal shares.
    """
    tried: dict[Route, tuple[float, float]] = {}
    for building in ROUTE_GRID:
        for object_ in ROUTE_GRID:
            if tied and building != object_:
                continue
            route = Route(building, object_, hard)
            tried[route] = mean_over_sets([rank_routed(s, route) for s in sets])
    order = sorted(tried, key=lambda r: (abs(r.building - r.object), abs(r.building - 0.5)))
    best = max(order, key=lambda r: tried[r])
    return best, tried[best]


def mean_over_sets(results: Sequence[ScoreResult]) -> tuple[float, float]:
    values = [per_query_values(r.per_query) for r in results]
    return (
        float(np.mean([v["hit@10"].mean() for v in values])),
        float(np.mean([v["map"].mean() for v in values])),
    )


@dataclass(frozen=True, slots=True)
class CrossFitted:
    """Hit@10 per set when each half's weights were picked on the other half."""

    router: tuple[tuple[str, float], ...]
    one_weight: tuple[tuple[str, float], ...]
    """The same check for G1 on the same grid (both weights equal)."""


def cross_fitted(
    sets: Mapping[str, RoutedSet], entity_of_query: Mapping[str, str], *, hard: bool
) -> CrossFitted:
    """Two folds by building or object: pick the weights on one fold, rank the other."""
    entities = sorted(set(entity_of_query.values()))
    random.Random(SEED).shuffle(entities)
    fold_of = {e: i % 2 for i, e in enumerate(entities)}

    def run(*, tied: bool) -> tuple[tuple[str, float], ...]:
        hits: dict[str, list[float]] = {s: [] for s in sets}
        for fold in (0, 1):
            masks = {
                s: np.asarray(
                    [fold_of[entity_of_query[q.query_id]] == fold for q in rs.queries], dtype=bool
                )
                for s, rs in sets.items()
            }
            train = [rs.subset(~masks[s]) for s, rs in sets.items()]
            route, _ = pick_route(train, hard=hard, tied=tied)
            for s, rs in sets.items():
                scored = rank_routed(rs.subset(masks[s]), route)
                hits[s].extend(per_query_values(scored.per_query)["hit@10"].tolist())
        return tuple((s, float(np.mean(v))) for s, v in hits.items())

    return CrossFitted(router=run(tied=False), one_weight=run(tied=True))


def _rows(scores: EntityScores, rows: NDArray[np.intp]) -> EntityScores:
    return EntityScores(
        query_ids=tuple(scores.query_ids[int(i)] for i in rows),
        entity_ids=scores.entity_ids,
        values=scores.values[rows],
        gallery_images=scores.gallery_images,
    )


# --- the whole run ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouterReport:
    result: RunResult
    guess: GuessReport
    route: Route
    cross: CrossFitted


def score_router(
    settings: Settings,
    *,
    runs: Sequence[str],
    guess_runs: Sequence[str],
    hard: bool,
    round_id: str,
    step: str,
    option: str,
    set_ids: Sequence[str],
    gallery: str,
    publishable_queries: bool = False,
    not_found: bool = False,
) -> RouterReport:
    """Train the guess, pick the route on the tune halves, score it and write one result."""
    if len(runs) < 2 or not set_ids:
        msg = "G4 needs two or more vector runs and at least one set"
        raise ValueError(msg)
    gallery_ids = read_gallery(settings, gallery)
    entity_of = {i.image_id: i.entity_id for i in read_images(settings.images_parquet)}
    type_of_image = photo_types(settings)
    models = tuple(load_model(settings, run, entity_of, gallery_ids) for run in runs)
    guessing = tuple(read_vectors(settings.vectors_path(r)) for r in guess_runs)
    report = guess_report(settings, guessing, gallery_ids, entity_of, type_of_image)

    def queries_of(set_id: str) -> list[Query]:
        loaded = load_queries(settings, set_id, "tune")
        if publishable_queries:
            return publishable_only(settings, loaded, in_gallery=gallery_ids)
        return loaded

    def routed_set(set_id: str) -> RoutedSet:
        queries = tuple(search_queries(queries_of(set_id)))
        return _routed_set(models, queries, report.guess, guessing)

    sets = {s: routed_set(s) for s in set_ids}
    route, (hit10, _) = pick_route(list(sets.values()), hard=hard)
    logger.info("route %s: mean hit@10 %.3f", route, hit10)
    entity_of_query = {q.query_id: q.entity_id for s in set_ids for q in queries_of(s)}
    cross = cross_fitted(sets, entity_of_query, hard=hard)
    scores = [
        set_score(
            rank_entities(routed(rs, route), rs.queries),
            test_set=set_id,
            half="tune",
            gallery=gallery,
        )
        for set_id, rs in sets.items()
    ]
    found: tuple[NotFoundScore, ...] = ()
    raw_found: NotFoundScore | None = None
    if not_found:
        raw_found, typed = _not_found(
            models,
            guess=report.guess,
            guessing=guessing,
            route=route,
            sets=sets,
            queries_of=queries_of,
            gallery=gallery,
        )
        found = (typed,)
    run_name = "+".join(runs)
    notes = _notes(report, route, cross, models, raw_found)
    result = write_result(
        eval_result_path(round_id, step, option),
        round_id=round_id,
        step=step,
        option=option,
        vector_run=run_name,
        query_vector_run=run_name,
        vectors=combined_info([m.vectors.info for m in models], notes),
        scores=scores,
        not_found=found,
        today=_today(),
    )
    return RouterReport(result=result, guess=report, route=route, cross=cross)


def photo_types(settings: Settings) -> dict[str, PhotoType]:
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
) -> GuessReport:
    queries = {s: load_queries(settings, s, "tune") for s in QUERY_SETS}
    left_out = {
        e for qs in queries.values() for q in qs for e in (q.entity_id, *q.expected_entity_ids)
    }
    labels = {
        i: type_of_image[i]
        for i in gallery_ids
        if i in type_of_image and entity_of[i] not in left_out
    }
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


def _routed_set(
    models: Sequence[Model],
    queries: Sequence[SearchQuery],
    guess: TypeGuess,
    guessing: Sequence[VectorSet],
) -> RoutedSet:
    parts = tuple(entity_scores(m.vectors, m.gallery, queries) for m in models)
    p = guess.p_building(features(guessing, [q.image_id for q in queries]))
    return RoutedSet(tuple(queries), parts, tuple(min_max(x.values) for x in parts), p)


def _not_found(
    models: Sequence[Model],
    *,
    guess: TypeGuess,
    guessing: Sequence[VectorSet],
    route: Route,
    sets: Mapping[str, RoutedSet],
    queries_of: Callable[[str], list[Query]],
    gallery: str,
) -> tuple[NotFoundScore, NotFoundScore]:
    """T8 with routed weights, scored twice: (raw, cut-off per guessed type).

    Raw: confidence is the routed mix of the top entity's cosines. Models' cosines sit on
    different scales, so routing moves each photo type to its own scale. Per type: each
    confidence becomes its rank among in-index confidences of the same guessed type (share
    at or below it), so one cut-off acts as a separate cut-off per type.
    """
    in_index = [
        sets[s]
        if s in sets
        else _routed_set(models, search_queries(queries_of(s)), guess, guessing)
        for s in IN_INDEX_SETS
    ]
    asked = not_found_queries(queries_of(NOT_FOUND_SET))
    missing = tuple(entity_scores(m.vectors, m.gallery, asked) for m in models)
    p_missing = guess.p_building(features(guessing, [q.image_id for q in asked]))
    found = [c for rs in in_index for c in _confidences(rs.parts, rs.p_building, route)]
    p_found = np.concatenate([rs.p_building for rs in in_index])
    lost = _confidences(missing, p_missing, route)

    def score(found: Sequence[Confidence], lost: Sequence[Confidence]) -> NotFoundScore:
        return not_found_score(
            found,
            lost,
            test_set=NOT_FOUND_SET,
            half="tune",
            gallery=gallery,
            in_index_sets=IN_INDEX_SETS,
        )

    found_scores = np.asarray([c.score for c in found])
    ladder = {side: np.sort(found_scores[(p_found > SWITCH_AT) == side]) for side in (True, False)}

    def per_type(scored: Sequence[Confidence], p: Float64s) -> list[Confidence]:
        out: list[Confidence] = []
        for c, building in zip(scored, p > SWITCH_AT, strict=True):
            same = ladder[bool(building)]
            share = np.searchsorted(same, c.score, side="right") / same.size
            out.append(Confidence(c.query_id, float(share)))
        return out

    return score(found, lost), score(per_type(found, p_found), per_type(lost, p_missing))


def _confidences(
    parts: Sequence[EntityScores], p_building: Float64s, route: Route
) -> tuple[Confidence, ...]:
    weights = model_weights(route.first_weights(p_building), len(parts))
    return confidences(fuse_per_query(parts, weights), parts, weights)


def _notes(
    report: GuessReport,
    route: Route,
    cross: CrossFitted,
    models: Sequence[Model],
    raw_found: NotFoundScore | None,
) -> str:
    first, rest = models[0].vectors.info.option, "+".join(m.vectors.info.option for m in models[1:])
    kind = (
        "hard switch at p > 0.5" if route.hard else "weight = p * w_building + (1 - p) * w_object"
    )
    accuracy = ", ".join(f"{s} {100 * a:.1f}% of {n}" for s, n, a in report.query_accuracy)
    cross_router = ", ".join(f"{s} {100 * h:.1f}" for s, h in cross.router)
    cross_g1 = ", ".join(f"{s} {100 * h:.1f}" for s, h in cross.one_weight)
    return (
        f"G4 routed weighted sum of {first} and {rest}: {kind}; {first} weight "
        f"{route.building:.1f} for buildings, {route.object:.1f} for objects, picked from the "
        f"grid by mean hit@10 over the scored tune halves. "
        f"Type guess: logistic regression on {'+'.join(report.guess.runs)} "
        f"(penalty {report.guess.penalty:g}), trained on {report.trained_on} gallery images, "
        f"{report.left_out_entities} query entities left out; held-out balanced accuracy "
        f"{100 * report.held_out_accuracy:.1f}%; right on tune queries: {accuracy}. "
        f"Two-fold check, hit@10 with weights picked on the other fold: G4 {cross_router}; "
        f"G1 on the same grid {cross_g1}"
        + (
            ""
            if raw_found is None
            else (
                f". T8 entry: cut-off per guessed type (confidence = rank among in-index "
                f"confidences of the same type). With one cut-off on the raw routed cosines: "
                f"wrong but confident {100 * raw_found.wrong_confident.value:.1f}% "
                f"[{100 * raw_found.wrong_confident.low:.1f}, "
                f"{100 * raw_found.wrong_confident.high:.1f}], matched at 5% wrong "
                f"{100 * raw_found.match_rate_at_target.value:.1f}% "
                f"[{100 * raw_found.match_rate_at_target.low:.1f}, "
                f"{100 * raw_found.match_rate_at_target.high:.1f}]"
            )
        )
    )


def _today() -> date:
    return datetime.now(UTC).date()
