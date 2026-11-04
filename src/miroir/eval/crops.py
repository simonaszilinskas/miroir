"""Scores of the crop slice test (`miroir.jobs.crop_slice`, docs/search-levels.md).

The machine sends back, per query, the cosine of every gallery row's whole vectors and the
best cosine among each stored file's crops, per grid and model, and a summary of the same
query searched over the whole `pop-all` index (`summarize_full`). Here they are ranked the
site's way (`miroir.web.ranking`): an entity takes the best of its images, per model; each
model's scores are min-max scaled over the entities and mixed with the weight of the bucket
the photo is routed to. With crops, an image's score is the best of its whole and crop scores.

Two measures:

1. **Slice.** The ranking over the gallery's entities only (about 50,000 images): easier than
   the full index, since most look-alikes are missing.
2. **Full-index estimate.** Every entity of `pop-all` with its whole-image scores, the
   gallery's entities also with their crops. Optimistic for crops: the other 2.2 million
   images get no crops, so none of them climbs past the answer the way it would in a full
   crop index. The slice's sample of others shows how often that happens.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from miroir.config import Settings, eval_result_path
from miroir.eval.scoring import first_rank, tie_ranks
from miroir.jobs.crop_slice import A7_SETS
from miroir.web import ranking

if TYPE_CHECKING:
    from miroir.eval.sets import Query
    from miroir.jobs.crop_slice import SliceSpec

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]
Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]

#: Entities kept per query from the full-index search (the answer is rarely further down,
#: and past it the estimate only says "not in the top 2,000").
FULL_TOP = 2000
HIT_KS = (1, 10)


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    grids: tuple[str, ...]
    """Grid names (`Grid.name`) whose crops count."""
    a7_crops: bool = False
    """EmbeddingGemma 2 crops count too (only on the sample that has them)."""


VARIANTS = (
    Variant("a-whole", ()),
    Variant("b-whole+2x2", ("g2",)),
    Variant("c-whole+2x2+3x3", ("g2", "g3")),
    Variant("d-whole+3x3", ("g3",)),
)
A7_VARIANTS = (
    Variant("a-whole", ()),
    Variant("b-whole+2x2", ("g2",), a7_crops=True),
    Variant("c-whole+2x2+3x3", ("g2", "g3"), a7_crops=True),
)


# --- entities ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Groups:
    """Rows grouped by entity, for a max per entity with `np.maximum.reduceat`."""

    entity_ids: tuple[str, ...]
    row_entity: Ints
    order: Ints
    starts: Ints
    present: Ints
    """Entity of each group, in group order."""

    @classmethod
    def of(cls, row_entities: Sequence[str]) -> Groups:
        entity_ids = tuple(sorted(set(row_entities)))
        row_of = {e: i for i, e in enumerate(entity_ids)}
        row_entity = np.fromiter((row_of[e] for e in row_entities), np.int64, len(row_entities))
        order = np.argsort(row_entity, kind="stable")
        ordered = row_entity[order]
        starts = np.flatnonzero(np.r_[True, np.diff(ordered) != 0]) if len(ordered) else ordered
        return cls(
            entity_ids, row_entity, order, starts, ordered[starts] if len(ordered) else ordered
        )

    @property
    def count(self) -> int:
        return len(self.entity_ids)

    def entity_max(self, values: Floats) -> Floats:
        """Best value per entity; -inf for an entity with no row."""
        out = np.full(self.count, -np.inf, dtype=np.float32)
        if len(self.order):
            out[self.present] = np.maximum.reduceat(values[self.order], self.starts)
        return out


def fused(e7: Floats, e3: Floats, a7_weight: float) -> Floats:
    """The site's mix (`ranking.rank_photo`): a model weighted 0 is left out."""
    return ranking.rank_photo(e7, e3, a7_weight).fused


# --- the full index, summarised on the machine --------------------------------------------


@dataclass(frozen=True, slots=True)
class FullScores:
    """One query over the whole index: each model's range over the entities, the best
    entities by the site's mix, and the gallery's entities."""

    lo7: float
    hi7: float
    lo3: float
    hi3: float
    top: Ints
    """Entity rows (of the full catalog), best first."""
    top_e7: Floats
    top_e3: Floats
    gallery_e7: Floats
    gallery_e3: Floats


def _range(values: Floats) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)


def summarize_full(
    e7: Floats, e3: Floats, a7_weight: float, gallery_rows: Ints, top: int = FULL_TOP
) -> FullScores:
    lo7, hi7 = _range(e7)
    lo3, hi3 = _range(e3)
    best = ranking.top_entities(fused(e7, e3, a7_weight), top)
    # Fewer entities than `top` may be ranked (MegaLoc scores only its shortlist): pad with
    # row -1 and -inf, so every query's arrays have one shape.
    pad = top - len(best)
    neg = np.full(pad, -np.inf, dtype=np.float32)
    return FullScores(
        lo7,
        hi7,
        lo3,
        hi3,
        np.concatenate([best, np.full(pad, -1, dtype=np.int64)]),
        np.concatenate([e7[best].astype(np.float32), neg]),
        np.concatenate([e3[best].astype(np.float32), neg]),
        e7[gallery_rows],
        e3[gallery_rows],
    )


def full_rank(
    full: FullScores,
    *,
    top_in_gallery: Ints,
    crop7: Floats,
    crop3: Floats,
    a7_weight: float,
    right: Sequence[int],
) -> int:
    """Rank of the first right gallery entity over the whole index, the gallery's entities
    taking their crop scores too (`crop7`, `crop3`: best crop per gallery entity, -inf when
    none). `top_in_gallery`: for each of `full.top`, its gallery entity row, or -1.

    The other entities keep their whole scores; one beyond `full.top` cannot pass the answer
    unless the answer falls below the last of `full.top`. Returns 0 when there is no right
    entity, and `len(full.top) + 1` or more when the answer is not reached."""
    if not right:
        return 0
    g7 = np.maximum(full.gallery_e7, crop7)
    g3 = np.maximum(full.gallery_e3, crop3)
    outside = top_in_gallery < 0
    e7 = np.concatenate([g7, full.top_e7[outside]])
    e3 = np.concatenate([g3, full.top_e3[outside]])
    signals: list[tuple[Floats, float, float, float]] = []
    for values, weight, low, high in (
        (e7, a7_weight, full.lo7, full.hi7),
        (e3, 1 - a7_weight, full.lo3, full.hi3),
    ):
        if weight <= 0:
            continue
        finite = values[np.isfinite(values)]
        lo = min(low, float(finite.min())) if finite.size else low
        hi = max(high, float(finite.max())) if finite.size else high
        signals.append((values, weight, lo, hi))
    total = np.zeros(len(e7), dtype=np.float64)
    alive = np.ones(len(e7), dtype=bool)
    for values, weight, lo, hi in signals:
        span = hi - lo if hi > lo else 1.0
        finite = np.isfinite(values)
        alive &= finite
        total += weight * np.where(finite, (values - lo) / span, 0.0)
    total[~alive] = -np.inf
    alive_right = [r for r in right if np.isfinite(total[r])]
    if not alive_right:
        return len(full.top) + 1
    return tie_ranks(total, alive_right)[0]


# --- the slice ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scores:
    """What the machine sent back, in the Mac's arrays (`miroir.jobs.crop_slice.load_scores`)."""

    query_ids: tuple[str, ...]
    a7_weight: Floats
    buckets: tuple[str, ...]
    row_entities: tuple[str, ...]
    row_file: Ints
    """Per gallery row, its column in the crop tables; -1 for a row with no stored file."""
    whole7: NDArray[np.float16]
    whole3: NDArray[np.float16]
    crops3: Mapping[str, NDArray[np.float16]]
    """Grid name -> (queries, files): best crop cosine of each file."""
    crops7: Mapping[str, NDArray[np.float16]]
    """The same for EmbeddingGemma 2; -inf for files without its crops."""
    a7_files: Bools
    """Per crop column: the file has EmbeddingGemma 2 crops."""
    done: Bools
    """Per crop column: the machine got to it."""
    full: Sequence[FullScores] | None = None
    full_entities: tuple[str, ...] = ()
    """Entity ids of the full catalog rows named in `full[*].top`, row by row."""


@dataclass(frozen=True, slots=True)
class QueryCase:
    query_id: str
    set_id: str
    kind: str
    hidden_rows: Ints
    right: tuple[str, ...]
    answer_only: bool = False
    """The right entity counts only through the rows `hidden_rows` leaves it (e.g. the one
    target photo of VET); its other rows in the index are not seen either."""


@dataclass(frozen=True, slots=True)
class Ranked:
    """Per variant: the rank of each query's answer (0: no answer left to find) and the
    entity ranked first."""

    ranks: dict[str, list[int]] = field(default_factory=dict[str, list[int]])
    first: dict[str, list[str]] = field(default_factory=dict[str, list[str]])


def _row_scores(
    scores: Scores, q: int, variant: Variant, rows: Ints, *, whole: bool
) -> tuple[Floats, Floats]:
    """(EmbeddingGemma 2, MegaLoc) score of each of `rows`: whole and the variant's crops,
    or crops alone (`whole=False`)."""
    neg = np.full(len(rows), -np.inf, dtype=np.float32)
    s7 = scores.whole7[q, rows].astype(np.float32) if whole else neg.copy()
    s3 = scores.whole3[q, rows].astype(np.float32) if whole else neg.copy()
    cols = scores.row_file[rows]
    has = cols >= 0
    for grid in variant.grids:
        c3 = neg.copy()
        c3[has] = scores.crops3[grid][q, cols[has]]
        s3 = np.maximum(s3, c3)
        if variant.a7_crops:
            c7 = neg.copy()
            c7[has] = scores.crops7[grid][q, cols[has]]
            s7 = np.maximum(s7, c7)
    return s7, s3


def rank_slice(
    scores: Scores,
    cases: Sequence[QueryCase],
    variants: Sequence[Variant],
    *,
    rows: Ints | None = None,
) -> Ranked:
    """Each case searched over the gallery rows `rows` (all by default) per variant."""
    picked = np.arange(len(scores.row_entities), dtype=np.int64) if rows is None else rows
    groups = Groups.of([scores.row_entities[r] for r in picked])
    position = {q: i for i, q in enumerate(scores.query_ids)}
    out = Ranked()
    for variant in variants:
        out.ranks[variant.name] = []
        out.first[variant.name] = []
    for case in cases:
        q = position[case.query_id]
        hidden = np.isin(picked, case.hidden_rows)
        right = [groups.entity_ids.index(e) for e in case.right if e in groups.entity_ids]
        for variant in variants:
            s7, s3 = _row_scores(scores, q, variant, picked, whole=True)
            s7[hidden] = -np.inf
            s3[hidden] = -np.inf
            e7, e3 = groups.entity_max(s7), groups.entity_max(s3)
            mix = fused(e7, e3, float(scores.a7_weight[q]))
            alive = [r for r in right if np.isfinite(e7[r])]
            out.ranks[variant.name].append(first_rank(mix, alive))
            top = ranking.top_entities(mix, 1)
            out.first[variant.name].append(groups.entity_ids[int(top[0])] if len(top) else "")
    return out


def rank_full(
    scores: Scores, cases: Sequence[QueryCase], variants: Sequence[Variant]
) -> dict[str, list[int]]:
    """Each case's full-index estimate per variant (`full_rank`)."""
    if scores.full is None:
        msg = "the machine sent no full-index scores"
        raise ValueError(msg)
    picked = np.arange(len(scores.row_entities), dtype=np.int64)
    groups = Groups.of(list(scores.row_entities))
    gallery_row = {e: i for i, e in enumerate(groups.entity_ids)}
    position = {q: i for i, q in enumerate(scores.query_ids)}
    out: dict[str, list[int]] = {v.name: [] for v in variants}
    for case in cases:
        q = position[case.query_id]
        full = scores.full[q]
        top_in_gallery = np.fromiter(
            (gallery_row.get(scores.full_entities[int(r)], -1) if r >= 0 else -1 for r in full.top),
            np.int64,
            len(full.top),
        )
        hidden = np.isin(picked, case.hidden_rows)
        right = [gallery_row[e] for e in case.right if e in gallery_row]
        if case.answer_only:
            full = _answer_only(scores, q, full, groups=groups, hidden=hidden, right=right)
        right = [r for r in right if np.isfinite(full.gallery_e7[r])]
        for variant in variants:
            c7, c3 = _row_scores(scores, q, variant, picked, whole=False)
            c7[hidden] = -np.inf
            c3[hidden] = -np.inf
            out[variant.name].append(
                full_rank(
                    full,
                    top_in_gallery=top_in_gallery,
                    crop7=groups.entity_max(c7),
                    crop3=groups.entity_max(c3),
                    a7_weight=float(scores.a7_weight[q]),
                    right=right,
                )
            )
    return out


def _answer_only(
    scores: Scores, q: int, full: FullScores, *, groups: Groups, hidden: Bools, right: list[int]
) -> FullScores:
    """`full` with the right entities' whole scores taken from their rows left in the
    gallery only. MegaLoc stays out for an entity the site's shortlist left out."""
    s7 = scores.whole7[q].astype(np.float32)
    s3 = scores.whole3[q].astype(np.float32)
    s7[hidden] = -np.inf
    s3[hidden] = -np.inf
    e7, e3 = groups.entity_max(s7), groups.entity_max(s3)
    g7, g3 = full.gallery_e7.copy(), full.gallery_e3.copy()
    for r in right:
        g7[r] = e7[r]
        g3[r] = e3[r] if np.isfinite(g3[r]) else -np.inf
    return replace(full, gallery_e7=g7, gallery_e3=g3)


# --- summaries -------------------------------------------------------------------------------


def hits(ranks: Sequence[int]) -> dict[str, float]:
    """hit@k over the queries with an answer to find, in percent, and how many there were."""
    found = np.asarray([r for r in ranks if r > 0])
    row: dict[str, float] = {"queries": float(len(found))}
    for k in HIT_KS:
        row[f"hit@{k}"] = round(100 * float((found <= k).mean()), 1) if found.size else 0.0
    return row


def look_alikes(before: Sequence[int], after: Sequence[int]) -> dict[str, int]:
    """Queries whose answer left first place (a wrong record now first), and the other way."""
    pairs = [(b, a) for b, a in zip(before, after, strict=True) if b > 0]
    return {
        "lost_first": sum(1 for b, a in pairs if b == 1 and a != 1),
        "gained_first": sum(1 for b, a in pairs if b != 1 and a == 1),
        "wrong_first_before": sum(1 for b, _ in pairs if b != 1),
        "wrong_first_after": sum(1 for _, a in pairs if a != 1),
    }


def by_group(
    cases: Sequence[QueryCase], ranks: Mapping[str, Sequence[int]], key: str
) -> dict[str, dict[str, dict[str, float]]]:
    """hit@k per variant for each value of `key` ("set_id" or "kind")."""
    groups: dict[str, list[int]] = {}
    for i, case in enumerate(cases):
        groups.setdefault(getattr(case, key), []).append(i)
    return {
        g: {v: hits([r[i] for i in idx]) for v, r in ranks.items()} for g, idx in groups.items()
    }


# --- reading what the machine sent -----------------------------------------------------------


def load_scores(folder: Path, spec: SliceSpec) -> Scores:
    """The score tables in `folder` (`miroir.jobs.crop_slice.fetch_outputs`)."""
    with np.load(folder / "queries.npz", allow_pickle=False) as data:
        query_ids = tuple(str(q) for q in data["query_ids"].tolist())
        a7_weight = np.asarray(data["a7_weight"], dtype=np.float32)
        buckets = tuple(str(b) for b in data["buckets"].tolist())
    with np.load(folder / "whole.npz", allow_pickle=False) as data:
        if tuple(str(i) for i in data["image_ids"].tolist()) != spec.gallery:
            msg = "whole.npz holds another gallery than the spec"
            raise ValueError(msg)
        row_entities = tuple(str(e) for e in data["entities"].tolist())
        whole7, whole3 = data["whole7"], data["whole3"]
    with np.load(folder / "crops.npz", allow_pickle=False) as data:
        files = [str(f) for f in data["files"].tolist()]
        grids = [g.name for g in spec.grids]
        crops3 = {g: data[f"a3_{g}"] for g in grids}
        crops7 = {g: data[f"a7_{g}"] for g in grids}
        a7_files = np.asarray(data["a7_files"], dtype=bool)
        done = np.asarray(data["done"], dtype=bool)
    column = {f: i for i, f in enumerate(files)}
    row_file = np.fromiter((column.get(f, -1) if f else -1 for f in spec.files), np.int64)
    full: list[FullScores] | None = None
    entities: tuple[str, ...] = ()
    path = folder / "full.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            named = {
                int(r): str(e)
                for r, e in zip(data["named_rows"], data["named_entities"], strict=True)
            }
            gallery = tuple(str(e) for e in data["gallery_entities"].tolist())
            if gallery != Groups.of(list(row_entities)).entity_ids:
                msg = "full.npz holds another list of gallery entities than whole.npz"
                raise ValueError(msg)
            top = np.asarray(data["top"], dtype=np.int64)
            # Full-catalog rows become positions in `entities`.
            order = sorted(named)
            entities = tuple(named[r] for r in order)
            position = {r: i for i, r in enumerate(order)}
            full = [
                FullScores(
                    *(float(x) for x in data["range"][q]),
                    top=np.asarray([position.get(int(r), -1) for r in top[q]], dtype=np.int64),
                    top_e7=data["top_e7"][q],
                    top_e3=data["top_e3"][q],
                    gallery_e7=data["gallery_e7"][q],
                    gallery_e3=data["gallery_e3"][q],
                )
                for q in range(len(query_ids))
            ]
    return Scores(
        query_ids=query_ids,
        a7_weight=a7_weight,
        buckets=buckets,
        row_entities=row_entities,
        row_file=row_file,
        whole7=whole7,
        whole3=whole3,
        crops3=crops3,
        crops7=crops7,
        a7_files=a7_files,
        done=done,
        full=full,
        full_entities=entities,
    )


def cases_of(spec: SliceSpec, queries: Sequence[Query]) -> list[QueryCase]:
    row = {image_id: i for i, image_id in enumerate(spec.gallery)}
    kind = {q.query_id: q.kind for q in queries}
    return [
        QueryCase(
            query_id=q.query_id,
            set_id=q.set_id,
            kind=kind.get(q.query_id, ""),
            hidden_rows=np.asarray([row[i] for i in q.hidden if i in row], dtype=np.int64),
            right=q.expected,
        )
        for q in spec.queries
    ]


def usable_rows(scores: Scores, *, a7: bool = False) -> Ints:
    """Gallery rows whose crops the machine made (and, with `a7`, with EmbeddingGemma 2 too).
    Rows with no stored file (Commons photos) count, whole only, unless `a7`."""
    has = scores.row_file >= 0
    ok = np.zeros(len(scores.row_file), dtype=bool)
    ok[has] = scores.done[scores.row_file[has]]
    if a7:
        ok[has] &= scores.a7_files[scores.row_file[has]]
    else:
        ok |= ~has
    return np.flatnonzero(ok).astype(np.int64)


# --- the report --------------------------------------------------------------------------------


def report(scores: Scores, cases: Sequence[QueryCase]) -> dict[str, object]:
    """Every measure of the test, ready for JSON. Cases with `answer_only` are measured
    apart (`vet_target_photo`)."""
    target = [c for c in cases if c.answer_only]
    cases = [c for c in cases if not c.answer_only]
    rows = usable_rows(scores)
    main = rank_slice(scores, cases, VARIANTS, rows=rows)
    out: dict[str, object] = {
        "gallery_rows": len(rows),
        "gallery_entities": Groups.of([scores.row_entities[int(r)] for r in rows]).count,
        "slice": {
            **by_group(cases, main.ranks, "set_id"),
            **{
                f"PARTS {k}": v
                for k, v in by_group(
                    [c for c in cases if c.set_id == "PARTS"],
                    {
                        n: [r for c, r in zip(cases, rs, strict=True) if c.set_id == "PARTS"]
                        for n, rs in main.ranks.items()
                    },
                    "kind",
                ).items()
            },
        },
    }
    look: dict[str, dict[str, dict[str, int]]] = {}
    for set_id in ("T2", "T3", "PARTS", "VET"):
        idx = [i for i, c in enumerate(cases) if c.set_id == set_id]
        base = [main.ranks["a-whole"][i] for i in idx]
        look[set_id] = {
            v.name: look_alikes(base, [main.ranks[v.name][i] for i in idx]) for v in VARIANTS[1:]
        }
    out["look_alikes"] = look
    out["buckets"] = {
        set_id: dict(
            Counter(
                scores.buckets[scores.query_ids.index(c.query_id)]
                for c in cases
                if c.set_id == set_id
            )
        )
        for set_id in sorted({c.set_id for c in cases})
    }
    a7_rows = usable_rows(scores, a7=True)
    a7_cases = [c for c in cases if c.set_id in A7_SETS]
    if len(a7_rows):
        sub = rank_slice(scores, a7_cases, A7_VARIANTS, rows=a7_rows)
        a3_only = rank_slice(scores, a7_cases, VARIANTS[1:3], rows=a7_rows)
        out["a7_sample"] = {
            "gallery_rows": len(a7_rows),
            "both_models": by_group(a7_cases, sub.ranks, "set_id"),
            "megaloc_crops_only": by_group(a7_cases, a3_only.ranks, "set_id"),
        }
    if scores.full is not None:
        full = rank_full(scores, cases, VARIANTS)
        out["full_index_estimate"] = by_group(cases, full, "set_id")
        out["full_ranks"] = {
            c.query_id: {v: full[v][i] for v in full}
            for i, c in enumerate(cases)
            if c.set_id == "VET"
        }
    if target:
        alone = rank_slice(scores, target, VARIANTS, rows=rows)
        out["vet_target_photo"] = {
            c.query_id: {v: alone.ranks[v][i] for v in alone.ranks} for i, c in enumerate(target)
        }
        if scores.full is not None:
            far = rank_full(scores, target, VARIANTS)
            out["vet_target_photo_full"] = {
                c.query_id: {v: far[v][i] for v in far} for i, c in enumerate(target)
            }
    out["vet_ranks"] = {
        c.query_id: {v: main.ranks[v][i] for v in main.ranks}
        for i, c in enumerate(cases)
        if c.set_id == "VET"
    }
    return out


def target_photo_cases(
    scores: Scores,
    spec: SliceSpec,
    cases: Sequence[QueryCase],
    *,
    set_id: str,
    ref_of: Mapping[str, str],
    ref: str,
) -> list[QueryCase]:
    """`set_id`'s cases again, their answer seen only through the photos of record `ref`
    (VET: APMH047981, the photo from before 1896): the entity's other rows are hidden."""
    out: list[QueryCase] = []
    for case in cases:
        if case.set_id != set_id:
            continue
        others = [
            row
            for row, (image_id, entity) in enumerate(
                zip(spec.gallery, scores.row_entities, strict=True)
            )
            if entity in case.right and ref_of.get(image_id) != ref
        ]
        hidden = np.union1d(case.hidden_rows, np.asarray(others, dtype=np.int64))
        out.append(replace(case, hidden_rows=hidden.astype(np.int64), answer_only=True))
    return out


def score_slice(settings: Settings) -> Path:
    """Score the crop slice test from the tables `miroir scw crop-slice` brought back; write
    eval/results/R3/SL-crop-slice.json and return its path."""
    from miroir.eval.parts import VET, VET_TARGET_REF  # noqa: PLC0415
    from miroir.jobs.crop_slice import read_spec, slice_dir, slice_queries, vet_refs  # noqa: PLC0415

    folder = slice_dir(settings)
    spec = read_spec((folder / "spec.json").read_bytes())
    scores = load_scores(folder, spec)
    cases = cases_of(spec, slice_queries(settings))
    refs = vet_refs(settings, spec)
    target = target_photo_cases(scores, spec, cases, set_id=VET, ref_of=refs, ref=VET_TARGET_REF)
    path = eval_result_path("R3", "SL", "crop-slice")
    path.write_text(
        json.dumps(report(scores, [*cases, *target]), indent=2, ensure_ascii=False) + "\n"
    )
    return path
