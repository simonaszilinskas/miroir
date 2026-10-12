"""Steps H1 and H3: search records with a vision model's answer, alone or mixed with images.

H1: the model's answer about a query photo (`miroir.jobs.describe`) becomes a text query,
searched with BM25 over the gallery's records. A record's score goes to each of its gallery
images; the query's hidden images leave the gallery, as in `scoring.entity_scores`, and an
entity scores its best image. So text and image runs rank the same entities and can be fused.

H3 (a preview of G3): the H1 scores mixed with an image model's scores, G1-style (weighted sum
of min-max scores), with one weight for every set, picked on the tune halves.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np

from miroir.config import Settings, eval_result_path
from miroir.eval.bm25 import Bm25, fold, record_text, tokens
from miroir.eval.fusion import Method, fuse, split_weight
from miroir.eval.results import RunResult, set_score, write_result
from miroir.eval.run import publishable_only, search_queries
from miroir.eval.scoring import (
    EntityScores,
    Floats,
    Gallery,
    SearchQuery,
    entity_scores,
    per_query_values,
    rank_entities,
)
from miroir.eval.sets import load_queries, read_gallery
from miroir.eval.vectors import Route, RunInfo, read_vectors
from miroir.images.fetch import LONG_SIDE
from miroir.jobs.describe import (
    Answer,
    DescribeOption,
    Description,
    parse_description,
    read_answers,
)
from miroir.records.model import Record
from miroir.records.store import read_records

logger = logging.getLogger(__name__)

#: Image-model weights tried for H3; the text side gets the rest.
H3_WEIGHT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: Confidence bands for the report: [low, high).
CONFIDENCE_BANDS = ((0.0, 0.5), (0.5, 0.8), (0.8, 1.01))


def query_text(description: Description | None, *, visible_text: bool = True) -> str:
    """The text searched for one answer. Name and commune count twice: they pin a record."""
    if description is None:
        return ""
    d = description
    parts = [d.name, d.name, d.kind, d.commune, d.commune, d.departement, d.region, d.period]
    parts += [d.author, d.description]
    if visible_text:
        parts.append(d.visible_text)
    return " ".join(p for p in parts if p)


@dataclass(frozen=True, slots=True)
class TextGallery:
    """The gallery's images, each with its entity and the row of its record in the index."""

    image_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    record_rows: np.ndarray[tuple[int], np.dtype[np.int64]]
    index: Bm25


def text_gallery(records: Sequence[Record], image_ids: Sequence[str]) -> TextGallery:
    """BM25 over the records that have an image in `image_ids`."""
    record_of = {i.image_id: r for r in records for i in r.images}
    image_of = {i.image_id: i for r in records for i in r.images}
    missing = [i for i in image_ids if i not in record_of]
    if missing:
        msg = f"{len(missing)} gallery image(s) have no record, e.g. {missing[:3]}"
        raise ValueError(msg)
    row_of: dict[str, int] = {}
    texts: list[str] = []
    for image_id in image_ids:
        record = record_of[image_id]
        if record.ref not in row_of:
            row_of[record.ref] = len(texts)
            texts.append(record_text(record))
    rows = np.fromiter((row_of[record_of[i].ref] for i in image_ids), dtype=np.int64)
    entities = tuple(image_of[i].entity_id for i in image_ids)
    logger.info("BM25 over %d records (%d images)", len(texts), len(image_ids))
    return TextGallery(tuple(image_ids), entities, rows, Bm25(texts))


def text_entity_scores(
    gallery: TextGallery, queries: Sequence[SearchQuery], texts: Mapping[str, str]
) -> EntityScores:
    """BM25 score of every entity for every query, from the text asked about its image."""
    names, codes = np.unique(np.asarray(gallery.entity_ids), return_inverse=True)
    order = np.argsort(codes, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(codes[order]) != 0])
    row_of = {image_id: i for i, image_id in enumerate(gallery.image_ids)}
    out: Floats = np.empty((len(queries), len(names)), dtype=np.float32)
    for j, query in enumerate(queries):
        per_image: Floats = gallery.index.scores(texts.get(query.image_id, ""))[gallery.record_rows]
        hidden = [row_of[i] for i in (query.image_id, *query.excluded_image_ids) if i in row_of]
        per_image[hidden] = -np.inf
        out[j] = np.maximum.reduceat(per_image[order], starts)
    return EntityScores(
        query_ids=tuple(q.query_id for q in queries),
        entity_ids=tuple(str(e) for e in names),
        values=out,
        gallery_images=len(gallery.image_ids),
    )


@dataclass(frozen=True, slots=True)
class _Set:
    set_id: str
    queries: tuple[SearchQuery, ...]
    text: EntityScores
    text_without_visible: EntityScores


@dataclass(frozen=True, slots=True)
class Prepared:
    """Everything H1 and H3 need: answers, records, and per-set text scores."""

    option: DescribeOption
    gallery: str
    gallery_ids: tuple[str, ...]
    answers: Mapping[str, Answer]
    descriptions: Mapping[str, Description | None]
    records: tuple[Record, ...]
    sets: tuple[_Set, ...]


def prepare(
    settings: Settings, option: DescribeOption, set_ids: Sequence[str], gallery: str
) -> Prepared:
    records = tuple(read_records(settings.records_parquet, settings.images_parquet))
    gallery_ids = tuple(read_gallery(settings, gallery))
    answers = read_answers(settings.vlm_answers_path(option.slug))
    descriptions = {i: parse_description(a.text) for i, a in answers.items()}
    unreadable = sum(1 for d in descriptions.values() if d is None)
    if unreadable:
        logger.warning("%s: %d answer(s) hold no JSON object", option.slug, unreadable)
    texts = {i: query_text(d) for i, d in descriptions.items()}
    plain = {i: query_text(d, visible_text=False) for i, d in descriptions.items()}
    searched = text_gallery(records, gallery_ids)
    sets: list[_Set] = []
    for set_id in set_ids:
        loaded = publishable_only(
            settings, load_queries(settings, set_id, "tune"), in_gallery=gallery_ids
        )
        queries = tuple(search_queries(loaded))
        unanswered = sum(1 for q in queries if q.image_id not in answers)
        if unanswered:
            logger.warning("%s: %d of %d queries have no answer", set_id, unanswered, len(queries))
        sets.append(
            _Set(
                set_id,
                queries,
                text_entity_scores(searched, queries, texts),
                text_entity_scores(searched, queries, plain),
            )
        )
    return Prepared(option, gallery, gallery_ids, answers, descriptions, records, tuple(sets))


def score_h1(prepared: Prepared, *, round_id: str) -> RunResult:
    """Rank entities by the BM25 score of the model's answer; write H-H1-<slug>.json."""
    p = prepared
    scores = [
        set_score(
            rank_entities(s.text, s.queries), test_set=s.set_id, half="tune", gallery=p.gallery
        )
        for s in p.sets
    ]
    return write_result(
        eval_result_path(round_id, "H", f"H1-{p.option.slug}"),
        round_id=round_id,
        step="H",
        option=f"H1-{p.option.slug}",
        vector_run=_answers_label(p.option),
        query_vector_run=_answers_label(p.option),
        vectors=_h1_info(p),
        scores=scores,
        today=_today(),
    )


def score_h3(
    settings: Settings, prepared: Prepared, image_run: str, image_label: str, *, round_id: str
) -> tuple[RunResult, float]:
    """Mix H1 with an image run (G1 weighted sum); one image weight picked over all sets.

    Returns the result and the image weight chosen.
    """
    p = prepared
    image_parts = image_scores(settings, image_run, p)
    tried: dict[float, float] = {}
    for weight in H3_WEIGHT_GRID:
        hits = [
            per_query_values(rank_entities(_mix(img, s.text, weight), s.queries).per_query)
            for img, s in zip(image_parts, p.sets, strict=True)
        ]
        tried[weight] = float(np.mean([h["hit@10"].mean() for h in hits]))
        logger.info("H3 image weight %.1f: mean hit@10 %.3f", weight, tried[weight])
    best = max(sorted(H3_WEIGHT_GRID, key=lambda w: abs(w - 0.5)), key=lambda w: tried[w])
    scores = [
        set_score(
            rank_entities(_mix(img, s.text, best), s.queries),
            test_set=s.set_id,
            half="tune",
            gallery=p.gallery,
        )
        for img, s in zip(image_parts, p.sets, strict=True)
    ]
    image_info = read_vectors(settings.vectors_path(image_run)).info
    grid = ", ".join(f"{w:.1f}: {100 * h:.1f}" for w, h in tried.items())
    info = RunInfo(
        option=f"{image_info.option}+H1",
        model=f"{image_info.model} + {p.option.model}",
        resolution=max(image_info.resolution, LONG_SIDE),
        pooling="fusion",
        gallery=p.gallery,
        created=_today(),
        cost_usd=image_info.cost_usd + _cost(p),
        wall_time_s=image_info.wall_time_s,
        route=Route.HOSTED,
        notes=(
            f"G1 mix of {image_info.option} {best:.1f} and H1 BM25 text {1 - best:.1f}; image "
            f"weight picked by mean hit@10 over {', '.join(s.set_id for s in p.sets)} "
            f"(tune halves; {grid})"
        ),
    )
    option = f"H3-{image_label}+H1-{p.option.slug}"
    result = write_result(
        eval_result_path(round_id, "H", option),
        round_id=round_id,
        step="H",
        option=option,
        vector_run=f"{image_run}+{_answers_label(p.option)}",
        query_vector_run=f"{image_run}+{_answers_label(p.option)}",
        vectors=info,
        scores=scores,
        today=_today(),
    )
    return result, best


def image_scores(settings: Settings, image_run: str, prepared: Prepared) -> list[EntityScores]:
    """An image run's entity scores on each prepared set, over the same gallery."""
    images = {i.image_id: i.entity_id for r in prepared.records for i in r.images}
    vectors = read_vectors(settings.vectors_path(image_run))
    gallery = Gallery.from_vectors(vectors, images, prepared.gallery_ids)
    return [entity_scores(vectors, gallery, s.queries) for s in prepared.sets]


def _mix(image: EntityScores, text: EntityScores, image_weight: float) -> EntityScores:
    return fuse([image, text], split_weight(image_weight, 2), Method.WEIGHTED_SUM)


# --- report ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Band:
    label: str
    queries: int
    hit_at_10: float


@dataclass(frozen=True, slots=True)
class SetReport:
    set_id: str
    queries: int
    answered: int
    """Queries with an answer holding a JSON object."""
    named: int
    """Answers that give a name."""
    name_right: int
    """Half or more of the name's words appear in a right record's title or description."""
    commune_given: int
    commune_right: int
    """The model's commune is a right record's commune (only queries whose records have one)."""
    with_commune: int
    """Queries whose right records hold a commune."""
    commune_read: int
    """Right communes the model also read in the image's text (a caption, a label)."""
    kind_right: int
    """A word of the model's "type" appears in a right record's title or denomination."""
    visible_text: int
    """Answers that read some text in the image."""
    text_helped: int
    """In the top 10 with the visible text, not without it."""
    text_hurt: int
    bands: tuple[Band, ...]
    """Hit@10 of H1 by the model's stated confidence; "none" when it gave none."""


def report(prepared: Prepared) -> list[SetReport]:
    """How often the model named the place or object, read text, and was right when sure."""
    by_entity: dict[str, list[Record]] = defaultdict(list)
    for record in prepared.records:
        by_entity[record.entity_id].append(record)
    out: list[SetReport] = []
    for s in prepared.sets:
        with_text = rank_entities(s.text, s.queries).per_query
        without = rank_entities(s.text_without_visible, s.queries).per_query
        hit = {q.query_id: q.rank <= 10 for q in with_text}
        hit_plain = {q.query_id: q.rank <= 10 for q in without}
        counts: dict[str, int] = defaultdict(int)
        banded: dict[str, list[bool]] = defaultdict(list)
        for query in s.queries:
            d = prepared.descriptions.get(query.image_id)
            right = [r for e in query.entity_ids for r in by_entity[e]]
            communes = {fold(r.commune) for r in right if r.commune}
            counts["with_commune"] += bool(communes)
            banded[_band(d)].append(hit[query.query_id])
            if d is None:
                continue
            counts["answered"] += 1
            words = set(tokens(" ".join(f"{r.title} {r.description}" for r in right)))
            named = tokens(d.name)
            if named:
                counts["named"] += 1
                counts["name_right"] += sum(w in words for w in named) >= len(named) / 2
            if d.commune:
                counts["commune_given"] += 1
                right_commune = fold(d.commune).strip() in communes
                counts["commune_right"] += right_commune
                counts["commune_read"] += right_commune and fold(d.commune) in fold(d.visible_text)
            kinds = set(tokens(" ".join(f"{r.title} {' '.join(r.denomination)}" for r in right)))
            counts["kind_right"] += any(w in kinds for w in tokens(d.kind))
            if d.visible_text:
                counts["visible_text"] += 1
                counts["text_helped"] += hit[query.query_id] and not hit_plain[query.query_id]
                counts["text_hurt"] += hit_plain[query.query_id] and not hit[query.query_id]
        bands = tuple(
            Band(label, len(v), float(np.mean(v))) for label, v in sorted(banded.items()) if v
        )
        out.append(
            SetReport(
                set_id=s.set_id,
                queries=len(s.queries),
                bands=bands,
                **{k: counts[k] for k in _COUNT_FIELDS},
            )
        )
    return out


_COUNT_FIELDS = (
    "answered",
    "named",
    "name_right",
    "commune_given",
    "commune_right",
    "with_commune",
    "commune_read",
    "kind_right",
    "visible_text",
    "text_helped",
    "text_hurt",
)


def _band(description: Description | None) -> str:
    if description is None or description.confidence is None:
        return "none"
    for low, high in CONFIDENCE_BANDS:
        if low <= description.confidence < high:
            return f"{low:.1f}-{min(high, 1.0):.1f}"
    return "none"


def _answers_label(option: DescribeOption) -> str:
    return f"vlm-answers-{option.slug}"


def _cost(prepared: Prepared) -> float:
    return sum(a.cost_usd for a in prepared.answers.values())


def _h1_info(prepared: Prepared) -> RunInfo:
    asked = len(prepared.answers)
    return RunInfo(
        option="H1",
        model=prepared.option.model,
        resolution=LONG_SIDE,
        pooling="bm25",
        gallery=prepared.gallery,
        created=_today(),
        cost_usd=_cost(prepared),
        wall_time_s=0.0,
        route=Route.OPENROUTER,
        notes=(
            f"{asked} answers in French JSON; BM25 over record title, description, "
            "denomination, domain, authors, period and place; query = name x2, type, "
            "commune x2, department, region, period, author, description, visible text"
        ),
    )


def _today() -> date:
    return datetime.now(UTC).date()
