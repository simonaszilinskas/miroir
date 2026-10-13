"""Step H2: what the phone's position, or a place the person names, buys the ranking.

Every set is ranked with the routed A7+A3 stack of R2 (G5 with F2 photos: one A7 weight per
bucket, MegaLoc the rest, min-max per model). A *policy* then changes, per query, which
entities may answer (a mask) and adds a bonus to some (a prior), before ranking:

- `none`: the stack as it is;
- `hard-2km`: only entities placed within 2 km of where the photo was taken;
- `soft-<b>-<s>km`: every entity stays; a placed one gains `b * exp(-d / s)`, d its distance
  in km from the photo, added to the fused score (which runs from 0 to 1);
- `known-<field>`: only entities sharing the answer's département, commune or museum, as if
  the person had picked it in a filter (those without a value stay in).

A mask applies before the min-max scaling, as the site's filters do. A query whose right
entity the mask removes counts as missed (rank past the last entity), never as skipped.

Position policies are scored on the queries whose photo has a position (EXIF GPS or the
Commons file page's camera location; `geo.queries`). Noise moves that position a fixed
distance in a seeded random direction. The "not found" side compares the routed raw cosine
of the top entity on T8c photos (no answer in the index) with photos that have one.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
from numpy.typing import NDArray

from miroir.config import SEED, Settings, eval_result_path
from miroir.eval.fusion import min_max
from miroir.eval.notfound import Confidence, NotFoundScore, not_found_score
from miroir.eval.results import RunResult, SetScore, set_score, write_result
from miroir.eval.router import QUERY_SETS, features, guess_report, photo_types
from miroir.eval.router3 import Bucket, asking_run, bucket_of, query_chroma
from miroir.eval.run import entity_map, publishable_only, search_queries
from miroir.eval.scoring import (
    Asked,
    Difference,
    EntityScores,
    Gallery,
    NotFoundQuery,
    QueryScore,
    ScoreResult,
    average_precision,
    entity_scores,
    paired_bootstrap,
    per_query_values,
)
from miroir.eval.sets import Query, load_queries, read_gallery
from miroir.eval.stack import combined_info
from miroir.eval.vectors import RunInfo, VectorSet, read_vectors
from miroir.geo.model import distance_km, moved
from miroir.records.mappers import PlaceHint
from miroir.records.model import Record
from miroir.records.store import read_places, read_positions, read_records

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]
Float64s = NDArray[np.float64]
Bools = NDArray[np.bool_]

#: The G5 + F2 weights of R2 (results.md, F2): A7 weight per bucket, A3 the rest.
G5_F2_WEIGHTS: Mapping[Bucket, float] = {
    Bucket.ARCHIVE_BUILDING: 1.0,
    Bucket.COLOUR_BUILDING: 0.0,
    Bucket.OBJECT: 0.2,
}
#: Radius of the hard filter, in km (experiments.md, H2).
RADIUS_KM = 2.0
#: Radii at which the report counts gallery entities around the photos.
RADII_KM = (0.5, 1.0, 2.0, 5.0, 10.0)
#: Grid of the soft prior: bonus at distance 0, and the distance over which it falls by e.
BONUSES = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
SCALES_KM = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
#: GPS errors simulated, in km.
NOISE_KM = (0.05, 0.3)
#: Fields a person could name: (policy name, entity attribute).
KNOWN_FIELDS = ("departement", "commune", "museum")


# --- one set, searched once ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Searched:
    """One set's queries with both models' raw entity scores and each query's A7 weight."""

    set_id: str
    query_ids: tuple[str, ...]
    query_images: tuple[str, ...]
    query_entities: tuple[str, ...]
    """The building or object each query shows (folds split on it)."""
    right: tuple[tuple[str, ...], ...]
    """Right entities per query; empty for a not-found set."""
    entity_ids: tuple[str, ...]
    e7: Floats
    e3: Floats
    a7_weight: Float64s
    gallery_images: int

    def subset(self, keep: Bools) -> Searched:
        rows = np.flatnonzero(keep)
        return Searched(
            set_id=self.set_id,
            query_ids=tuple(self.query_ids[i] for i in rows),
            query_images=tuple(self.query_images[i] for i in rows),
            query_entities=tuple(self.query_entities[i] for i in rows),
            right=tuple(self.right[i] for i in rows),
            entity_ids=self.entity_ids,
            e7=self.e7[rows],
            e3=self.e3[rows],
            a7_weight=self.a7_weight[rows],
            gallery_images=self.gallery_images,
        )

    @property
    def size(self) -> int:
        return len(self.query_ids)


@dataclass(frozen=True, slots=True)
class Change:
    """What a policy does to one set: entities each query may get, and a bonus per entity."""

    mask: Bools | None = None
    """(queries, entities); None keeps every entity."""
    bonus: Floats | None = None
    """(queries, entities), added after fusion; None adds nothing."""


def fused(searched: Searched, change: Change) -> Floats:
    """The routed weighted sum of min-max scores, after the policy's mask and bonus."""
    e7, e3 = searched.e7, searched.e3
    if change.mask is not None:
        e7 = np.where(change.mask, e7, -np.inf).astype(np.float32)
        e3 = np.where(change.mask, e3, -np.inf).astype(np.float32)
    w = searched.a7_weight[:, None].astype(np.float32)
    total = w * min_max(e7) + (1 - w) * min_max(e3)
    if change.bonus is not None:
        total = total + change.bonus
    total[~(np.isfinite(e7) & np.isfinite(e3))] = -np.inf
    return total.astype(np.float32)


def rank(searched: Searched, change: Change) -> ScoreResult:
    """Rank and AP of each query's right entities; a right entity masked out is a miss."""
    scores = fused(searched, change)
    code_of = {e: i for i, e in enumerate(searched.entity_ids)}
    worst = len(searched.entity_ids) + 1
    per_query: list[QueryScore] = []
    skipped = 0
    for j, right in enumerate(searched.right):
        codes = [code_of[e] for e in dict.fromkeys(right) if e in code_of]
        if not codes:
            skipped += 1
            continue
        row = scores[j]
        ranks = sorted(
            int((row > row[c]).sum() + (row[:c] == row[c]).sum()) + 1
            if np.isfinite(row[c])
            else worst
            for c in codes
        )
        found = [r for r in ranks if r < worst]
        ap = average_precision(found) * len(found) / len(ranks) if found else 0.0
        per_query.append(QueryScore(searched.query_ids[j], ranks[0], ap))
    return ScoreResult(
        per_query=tuple(per_query),
        skipped_no_answer=skipped,
        gallery_images=searched.gallery_images,
        gallery_entities=len(searched.entity_ids),
        ap_max_rank=None,
    )


#: Confidence of a search the policy left with no entity: below any cosine.
NOTHING_LEFT = -2.0


def confidences(searched: Searched, change: Change) -> tuple[Confidence, ...]:
    """Routed raw cosine of each query's top entity (`notfound`); `NOTHING_LEFT` when the
    policy removed every entity."""
    scores = fused(searched, change)
    top = np.argmax(scores, axis=1)
    rows = np.arange(searched.size)
    alive = np.isfinite(scores[rows, top])
    w = searched.a7_weight
    raw = w * searched.e7[rows, top] + (1 - w) * searched.e3[rows, top]
    value = np.where(alive, raw, NOTHING_LEFT)
    return tuple(Confidence(q, float(v)) for q, v in zip(searched.query_ids, value, strict=True))


# --- policies ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Places:
    """Entity positions in `Searched.entity_ids` order (NaN where unknown), and query ones."""

    lat: Float64s
    lon: Float64s
    query: Mapping[str, tuple[float, float]]
    """Position of each query image that has one."""

    def distances(self, searched: Searched, noise_km: float = 0.0, seed: int = SEED) -> Float64s:
        """(queries, entities) km from where each photo was taken; NaN for unplaced entities."""
        rng = random.Random(seed)
        out = np.empty((searched.size, len(self.lat)), dtype=np.float64)
        for j, image_id in enumerate(searched.query_images):
            lat, lon = self.query[image_id]
            bearing = rng.uniform(0.0, 2 * math.pi)
            if noise_km > 0:
                lat, lon = moved(lat, lon, noise_km, bearing)
            out[j] = distance_km(lat, lon, self.lat, self.lon)
        return out


def hard(distances: Float64s, radius_km: float, *, keep_unplaced: bool = True) -> Change:
    near = distances <= radius_km
    return Change(mask=near | np.isnan(distances) if keep_unplaced else near)


def soft(distances: Float64s, bonus: float, scale_km: float) -> Change:
    prior = bonus * np.exp(-np.nan_to_num(distances, nan=np.inf) / scale_km)
    return Change(bonus=prior.astype(np.float32))


def known(searched: Searched, value_of: Mapping[str, str]) -> Change:
    """Only entities with the query entity's value (or none); all when the query has none."""
    values = np.asarray([value_of.get(e, "") for e in searched.entity_ids], dtype=object)
    unknown = values == ""
    mask = np.ones((searched.size, len(values)), dtype=bool)
    for j, entity in enumerate(searched.query_entities):
        wanted = value_of.get(entity, "")
        if wanted:
            mask[j] = (values == wanted) | unknown
    return Change(mask=mask)


def entity_fields(
    records: Sequence[Record], hints: Mapping[str, PlaceHint]
) -> dict[str, dict[str, str]]:
    """Field name, then entity: its value (first non-empty over the entity's records)."""
    out: dict[str, dict[str, str]] = {f: {} for f in KNOWN_FIELDS}
    for record in records:
        hint = hints.get(record.ref)
        values = {
            "departement": record.departement or "",
            "commune": (hint.insee if hint and hint.insee else None)
            or (f"{record.departement}/{record.commune}" if record.commune else ""),
            "museum": hint.museum if hint and hint.museum else "",
        }
        for name, value in values.items():
            if value and record.entity_id not in out[name]:
                out[name][record.entity_id] = value
    return out


# --- choosing the soft prior, and the split-half check ---------------------------------------


def pick_soft(
    sets: Sequence[tuple[Searched, Float64s]], masks: Sequence[Bools] | None = None
) -> tuple[float, float]:
    """(bonus, scale) with the best mean hit@10 over sets (then mAP; then the smallest bonus)."""
    best: tuple[tuple[float, float, float], tuple[float, float]] | None = None
    for bonus in BONUSES:
        for scale in SCALES_KM:
            hit = ap = 0.0
            for k, (searched, d) in enumerate(sets):
                values = per_query_values(rank(searched, soft(d, bonus, scale)).per_query)
                keep = np.ones(searched.size, dtype=bool) if masks is None else masks[k]
                hit += float(values["hit@10"][keep].mean()) / len(sets)
                ap += float(values["map"][keep].mean()) / len(sets)
            key = (round(hit, 12), round(ap, 12), -bonus)
            if best is None or key > best[0]:
                best = (key, (bonus, scale))
    assert best is not None
    return best[1]


def soft_split_half(
    sets: Sequence[tuple[Searched, Float64s]], seed: int = SEED
) -> dict[str, float]:
    """Hit@10 per set when the prior was picked on the other half of the entities."""
    entities = sorted({e for s, _ in sets for e in s.query_entities})
    random.Random(seed).shuffle(entities)
    fold_of = {e: i % 2 for i, e in enumerate(entities)}
    folds = [np.asarray([fold_of[e] for e in s.query_entities]) for s, _ in sets]
    hits: dict[str, list[float]] = {s.set_id: [] for s, _ in sets}
    for fold in (0, 1):
        bonus, scale = pick_soft(sets, [f != fold for f in folds])
        for (searched, d), f in zip(sets, folds, strict=True):
            scored = per_query_values(rank(searched, soft(d, bonus, scale)).per_query)["hit@10"]
            hits[searched.set_id].extend(scored[f == fold].tolist())
    return {s: float(np.mean(v)) for s, v in hits.items()}


# --- the run -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyScore:
    option: str
    """Result file label: H2-<policy> or known-<field>."""
    description: str
    scores: Mapping[str, SetScore]
    differences: Mapping[str, Difference]
    """Set: paired hit@10 difference with `none` on the same queries."""
    not_found: NotFoundScore | None = None
    extra: str = ""


@dataclass(frozen=True, slots=True)
class LocationReport:
    gps_queries: Mapping[str, tuple[int, int]]
    """Set: (queries with a position, queries)."""
    placed: tuple[int, int]
    """(gallery entities with a place, gallery entities)."""
    nearby: Mapping[float, tuple[float, float]]
    """Radius: (median gallery entities within it, share of T8c photos with none)."""
    gps: tuple[PolicyScore, ...]
    known: tuple[PolicyScore, ...]
    soft_choice: tuple[float, float]
    soft_split: Mapping[str, float]
    results: tuple[RunResult, ...]


def score_location(
    settings: Settings,
    *,
    runs: Sequence[str],
    query_runs: Sequence[str],
    gallery: str,
    round_id: str,
    set_ids: Sequence[str] = ("T1", "T2", "T3", "T5", "T2b"),
    gps_sets: Sequence[str] = ("T2", "T2b"),
    not_found_set: str = "T8c",
    suffix: str = "",
) -> LocationReport:
    """Score every policy, write one result file each, and return the comparison.

    `suffix` ends every option label (e.g. "-full" for the full index).
    """
    searched, gallery_entities = _search(
        settings,
        runs=runs,
        query_runs=query_runs,
        gallery=gallery,
        set_ids=set_ids,
        not_found_set=not_found_set,
    )
    infos = [read_vectors(settings.vectors_path(r)).info for r in runs]
    write = _writer(runs, query_runs, gallery, round_id, infos)

    # Known fields, on every set.
    records = read_records(settings.records_parquet, settings.images_parquet)
    from miroir.geo.places import read_hints  # noqa: PLC0415 (reads the July exports)

    fields = entity_fields(records, read_hints(settings))
    base = {s: rank(searched[s], Change()) for s in set_ids}
    results: list[RunResult] = [
        write("known-none" + suffix, "No field known", {s: base[s] for s in set_ids}, None)
    ]
    known_scores: list[PolicyScore] = []
    for name in KNOWN_FIELDS:
        ranked = {s: rank(searched[s], known(searched[s], fields[name])) for s in set_ids}
        coverage = {
            s: sum(1 for e in searched[s].query_entities if fields[name].get(e)) for s in set_ids
        }
        extra = "; ".join(f"{s} {coverage[s]} of {searched[s].size} with a value" for s in set_ids)
        description = f"Gallery cut to entities with the answer's {name} (or none): {extra}"
        results.append(write(f"known-{name}{suffix}", description, ranked, None))
        known_scores.append(_policy(f"known-{name}", description, ranked, base, gallery, extra))

    # Positions, on the photos that have one.
    places = _places(settings, searched[set_ids[0]].entity_ids)
    located = {
        s: searched[s].subset(
            np.asarray([i in places.query for i in searched[s].query_images], dtype=np.bool_)
        )
        for s in (*gps_sets, not_found_set)
    }
    distances = {s: places.distances(located[s]) for s in located}
    in_index = [s for s in gps_sets if located[s].size]
    gps_base = {s: rank(located[s], Change()) for s in in_index}

    def not_found(changes: Mapping[str, Change]) -> NotFoundScore:
        found = [c for s in in_index for c in confidences(located[s], changes[s])]
        lost = confidences(located[not_found_set], changes[not_found_set])
        return not_found_score(
            found,
            lost,
            test_set=not_found_set,
            half="tune",
            gallery=gallery,
            in_index_sets=in_index,
        )

    policies: list[tuple[str, str, Callable[[Float64s], Change]]] = [
        ("H2-none", "No position used", lambda _d: Change()),
    ]
    policies.append(
        (
            f"H2-hard-{RADIUS_KM:g}km",
            f"Only placed entities within {RADIUS_KM:g} km of the photo",
            lambda d: hard(d, RADIUS_KM, keep_unplaced=False),
        )
    )
    sets_for_pick = [(located[s], distances[s]) for s in in_index if s == "T2"] or [
        (located[s], distances[s]) for s in in_index
    ]
    bonus, scale = pick_soft(sets_for_pick)
    split = soft_split_half(sets_for_pick)
    policies.append(
        (
            f"H2-soft-{bonus:g}-{scale:g}km",
            f"Bonus {bonus:g} * exp(-d / {scale:g} km) added to the fused score; picked on "
            f"T2 from bonus {BONUSES} x scale {SCALES_KM} km; split-half hit@10 "
            + ", ".join(f"{s} {100 * v:.1f}" for s, v in split.items()),
            lambda d: soft(d, bonus, scale),
        )
    )
    gps_scores: list[PolicyScore] = []
    for option, description, make in policies:
        changes = {s: make(distances[s]) for s in located}
        ranked = {s: rank(located[s], changes[s]) for s in in_index}
        nf = not_found(changes)
        results.append(write(option + suffix, description, ranked, nf))
        gps_scores.append(_policy(option, description, ranked, gps_base, gallery, nf=nf))

    # GPS errors, on the hard filter and the chosen prior.
    for noise in NOISE_KM:
        noisy = {s: places.distances(located[s], noise_km=noise) for s in located}
        noisy_policies: tuple[tuple[str, Callable[[Float64s], Change]], ...] = (
            (f"hard-{RADIUS_KM:g}km", lambda d: hard(d, RADIUS_KM, keep_unplaced=False)),
            (f"soft-{bonus:g}-{scale:g}km", lambda d: soft(d, bonus, scale)),
        )
        for label, make in noisy_policies:
            option = f"H2-{label}-noise{round(noise * 1000)}m"
            description = (
                f"As H2-{label}, the photo's position moved {round(noise * 1000)} m "
                "in a seeded random direction"
            )
            changes = {s: make(noisy[s]) for s in located}
            ranked = {s: rank(located[s], changes[s]) for s in in_index}
            nf = not_found(changes)
            results.append(write(option + suffix, description, ranked, nf))
            gps_scores.append(_policy(option, description, ranked, gps_base, gallery, nf=nf))

    nearby = {
        r: (
            float(np.median((distances["T2"] <= r).sum(axis=1))) if "T2" in distances else 0.0,
            float(((distances[not_found_set] <= r).sum(axis=1) == 0).mean())
            if located[not_found_set].size
            else 0.0,
        )
        for r in RADII_KM
    }
    return LocationReport(
        gps_queries={s: (located[s].size, searched[s].size) for s in located},
        placed=(int(np.isfinite(places.lat).sum()), gallery_entities),
        nearby=nearby,
        gps=tuple(gps_scores),
        known=tuple(known_scores),
        soft_choice=(bonus, scale),
        soft_split=split,
        results=tuple(results),
    )


def _policy(  # noqa: PLR0917 (one call per policy, read in order)
    option: str,
    description: str,
    ranked: Mapping[str, ScoreResult],
    base: Mapping[str, ScoreResult],
    gallery: str,
    extra: str = "",
    nf: NotFoundScore | None = None,
) -> PolicyScore:
    scores = {s: set_score(r, test_set=s, half="tune", gallery=gallery) for s, r in ranked.items()}
    differences = {
        s: next(d for d in paired_bootstrap(r.per_query, base[s].per_query) if d.metric == "hit@10")
        for s, r in ranked.items()
    }
    return PolicyScore(option, description, scores, differences, nf, extra)


def _writer(
    runs: Sequence[str],
    query_runs: Sequence[str],
    gallery: str,
    round_id: str,
    infos: Sequence[RunInfo],
) -> Callable[[str, str, Mapping[str, ScoreResult], NotFoundScore | None], RunResult]:
    weights = ", ".join(f"{b.value} {w:g}" for b, w in G5_F2_WEIGHTS.items())

    def write(
        option: str,
        description: str,
        ranked: Mapping[str, ScoreResult],
        nf: NotFoundScore | None,
    ) -> RunResult:
        notes = (
            f"{option}: {description}. Stack: G5 at the weights picked with F2 photos, "
            f"A7 weight per bucket ({weights}), A3 the rest, min-max per model; buckets as in G5"
        )
        return write_result(
            eval_result_path(round_id, "H", option),
            round_id=round_id,
            step="H",
            option=option,
            vector_run="+".join(runs),
            query_vector_run="+".join(query_runs),
            vectors=combined_info(infos, notes),
            scores=[
                set_score(r, test_set=s, half="tune", gallery=gallery) for s, r in ranked.items()
            ],
            today=_today(),
            not_found=() if nf is None else (nf,),
        )

    return write


def _places(settings: Settings, entity_ids: Sequence[str]) -> Places:
    by_entity = {p.entity_id: p for p in read_places(settings.places_parquet)}
    lat = np.asarray(
        [by_entity[e].latitude if e in by_entity else np.nan for e in entity_ids], dtype=np.float64
    )
    lon = np.asarray(
        [by_entity[e].longitude if e in by_entity else np.nan for e in entity_ids],
        dtype=np.float64,
    )
    query = {
        p.image_id: (p.latitude, p.longitude)
        for p in read_positions(settings.query_positions_parquet)
    }
    return Places(lat, lon, query)


def _search(
    settings: Settings,
    *,
    runs: Sequence[str],
    query_runs: Sequence[str],
    gallery: str,
    set_ids: Sequence[str],
    not_found_set: str,
) -> tuple[dict[str, Searched], int]:
    """Every set's tune queries (publishable, as in R2) searched once with both models."""
    if len(runs) != 2 or len(query_runs) != 2:
        msg = "give the A7 and A3 gallery runs, and their query runs, in that order"
        raise ValueError(msg)
    gallery_ids = read_gallery(settings, gallery)
    entity_of = entity_map(settings)
    vectors = [read_vectors(settings.vectors_path(r)) for r in runs]
    asked_runs = [read_vectors(settings.vectors_path(r)) for r in query_runs]
    # The guess leaves out the tune entities of the sets G5 was tuned on (results.md, G5).
    guess = guess_report(
        settings,
        (vectors[0],),
        gallery_ids,
        entity_of,
        photo_types(settings),
        query_sets=(*QUERY_SETS, "T2"),
    )
    loaded: dict[str, list[Query]] = {
        s: publishable_only(settings, load_queries(settings, s, "tune"), in_gallery=gallery_ids)
        for s in (*set_ids, not_found_set)
    }
    all_ids = [q.query_image_id for qs in loaded.values() for q in qs]
    asking = [asking_run(v, a, all_ids) for v, a in zip(vectors, asked_runs, strict=True)]
    chroma = query_chroma(settings, all_ids)
    p_of = dict(
        zip(
            asking[0].ids,
            guess.guess.p_building(features((asking[0],), asking[0].ids)).tolist(),
            strict=True,
        )
    )
    out: dict[str, Searched] = {}
    for set_id, qs in loaded.items():
        asked = (
            [NotFoundQuery(q.query_id, q.query_image_id, q.exclude_image_ids) for q in qs]
            if set_id == not_found_set
            else search_queries(qs)
        )
        parts = [
            chunked_entity_scores(a, v, entity_of, gallery_ids, asked)
            for a, v in zip(asking, vectors, strict=True)
        ]
        weights = np.asarray(
            [
                G5_F2_WEIGHTS[bucket_of(p_of[q.query_image_id], chroma[q.query_image_id])]
                for q in qs
            ],
            dtype=np.float64,
        )
        out[set_id] = Searched(
            set_id=set_id,
            query_ids=tuple(q.query_id for q in qs),
            query_images=tuple(q.query_image_id for q in qs),
            query_entities=tuple(q.entity_id for q in qs),
            right=tuple(q.expected_entity_ids for q in qs),
            entity_ids=parts[0].entity_ids,
            e7=parts[0].values,
            e3=parts[1].values,
            a7_weight=weights,
            gallery_images=parts[0].gallery_images,
        )
        logger.info("%s: %d queries searched", set_id, len(qs))
    return out, len(out[set_ids[0]].entity_ids)


#: Gallery images held in memory at once: the full index's MegaLoc vectors do not fit.
GALLERY_CHUNK = 16_384


def chunked_entity_scores(
    asking: VectorSet,
    run: VectorSet,
    entity_of: Mapping[str, str],
    gallery_ids: Sequence[str],
    queries: Sequence[Asked],
    *,
    chunk: int = GALLERY_CHUNK,
) -> EntityScores:
    """`scoring.entity_scores` over the gallery a chunk of images at a time (best per entity
    is a maximum, so the chunks combine exactly)."""
    names = sorted({entity_of[i] for i in gallery_ids})
    column = {e: j for j, e in enumerate(names)}
    out = np.full((len(queries), len(names)), -np.inf, dtype=np.float32)
    for start in range(0, len(gallery_ids), chunk):
        part = Gallery.from_vectors(run, entity_of, gallery_ids[start : start + chunk])
        scores = entity_scores(asking, part, queries)
        columns = np.asarray([column[e] for e in scores.entity_ids], dtype=np.int64)
        out[:, columns] = np.maximum(out[:, columns], scores.values)
    return EntityScores(
        query_ids=tuple(q.query_id for q in queries),
        entity_ids=tuple(names),
        values=out,
        gallery_images=len(gallery_ids),
    )


def _today() -> date:
    return datetime.now(UTC).date()
