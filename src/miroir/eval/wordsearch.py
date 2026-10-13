"""Step B on T7: word search scored with the website's own ranking, on its own index.

Each T7 tune query's words go through `miroir.web.engine.word_signals`, as on the site: the
EmbeddingGemma 2 text vector (embedded on Modal with the site's prompt, `TEXT_PROMPT`)
against every image vector of the index bundle, and BM25 over the bundle's record text. The
query's hidden images leave the index first. The two signals are then mixed:

- `bm25` or `a7` alone;
- a weighted sum of min-max scores (`ranking.fuse`, what the site does), A7 weight 0.5
  (the site before this step), 0.7 or 0.9.

Scores are kept per set label: T7 (French), T7en (English versions of some of the same
images), T7 restricted to those images (to compare languages on the same images), and T7
split by collection and by length (a few words or a sentence).

`refine` measures words that refine a photo search: T1, T2 and T5 tune photos, searched as
the site routes them, plus the commune of the right record as words, counting `refine`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.config import WEB_INDEX, Settings, eval_result_path
from miroir.eval.bm25 import Bm25
from miroir.eval.results import RunResult, SetScore, set_score, write_result
from miroir.eval.scoring import QueryScore, ScoreResult
from miroir.eval.sets import Query, load_queries
from miroir.eval.vectors import Route, RunInfo, VectorSet, read_vectors
from miroir.eval.words import EN_SET_ID, SET_ID, WRITER
from miroir.web import ranking
from miroir.web.bundle import INDEXES, entity_cosines
from miroir.web.catalog import CATALOG_FILE, Catalog, read_catalog
from miroir.web.engine import A3_FILE, A7_FILE, record_text, word_signals
from miroir.web.routing import GUESS_FILE, ROUTING_FILE, TextWeights, read_guess, read_routing

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]

#: A7 weights tried for the weighted sum; BM25 gets the rest.
WEIGHT_GRID = (0.5, 0.7, 0.9)
#: The site's weights before this step (half and half).
CURRENT_A7_WEIGHT = 0.5
#: A query of this many words or fewer counts as "words"; longer ones as a sentence.
SHORT_WORDS = 6
#: Weights tried for words refining a photo (the photo counts 1).
REFINE_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
REFINE_SETS = ("T1", "T2", "T5")
GALLERY_LABEL = f"web-{WEB_INDEX}"
#: Label of the text vector run (`miroir words embed`): T7, T7en and the refine communes.
TEXT_LABEL = "T7-T7en-places-tune-words"
QUERY_BATCH = 64


# --- the index -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WebIndex:
    """The website's bundle as search sees it: catalog, A7 image vectors, BM25 over records."""

    folder: Path
    catalog: Catalog
    a7: NDArray[np.float16]
    bm25: Bm25

    @classmethod
    def load(cls, settings: Settings) -> WebIndex:
        folder = settings.web_index_dir(WEB_INDEX)
        catalog = read_catalog(folder / CATALOG_FILE)
        a7: NDArray[np.float16] = np.load(folder / A7_FILE, mmap_mode="r")
        bm25 = Bm25([record_text(r) for r in catalog.records])
        logger.info(
            "index %s: %d images, %d entities", WEB_INDEX, a7.shape[0], catalog.entity_count
        )
        return cls(folder, catalog, a7, bm25)

    def hidden_mask(self, query: Query) -> NDArray[np.bool_]:
        mask = np.ones(len(self.catalog.images), dtype=bool)
        rows = [
            self.catalog.image_row[i]
            for i in (query.query_image_id, *query.exclude_image_ids)
            if i in self.catalog.image_row
        ]
        mask[rows] = False
        return mask


# --- one signal pair per query ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signals:
    """Per query: the A7 and BM25 scores of every entity, and the right entity's row."""

    query_ids: tuple[str, ...]
    a7: Floats
    bm25: Floats
    right: NDArray[np.int64]


def word_query_signals(index: WebIndex, queries: Sequence[Query], vectors: Floats) -> Signals:
    """Both word signals for each query, its hidden images out (the site's `word_signals`)."""
    catalog = index.catalog
    e7 = np.empty((len(queries), catalog.entity_count), dtype=np.float32)
    eb = np.empty_like(e7)
    both = TextWeights(a7=1.0, bm25=1.0)
    for start in range(0, len(queries), QUERY_BATCH):
        batch = vectors[start : start + QUERY_BATCH]
        cosines = _cosines(batch, index.a7)
        for j in range(len(batch)):
            q = queries[start + j]
            mask = index.hidden_mask(q)
            a7, bm25 = word_signals(catalog, cosines[j], index.bm25.scores(q.text), mask, both)
            e7[start + j], eb[start + j] = a7.per_entity, bm25.per_entity
    right = np.asarray(
        [catalog.entity_row[q.expected_entity_ids[0]] for q in queries], dtype=np.int64
    )
    return Signals(tuple(q.query_id for q in queries), e7, eb, right)


def _cosines(queries: Floats, gallery: NDArray[np.float16]) -> Floats:
    out = np.empty((len(queries), gallery.shape[0]), dtype=np.float32)
    step = 16_384
    for g in range(0, gallery.shape[0], step):
        block = np.asarray(gallery[g : g + step], dtype=np.float32)
        out[:, g : g + len(block)] = queries @ block.T
    return out


# --- mixing and ranking ----------------------------------------------------------------------

Mixer = Callable[[Floats, Floats], Floats]


def weighted(a7_weight: float) -> Mixer:
    """The site's mix: weighted sum of min-max scores; a zero weight drops the signal."""

    def mix(a7: Floats, bm25: Floats) -> Floats:
        signals = [
            s
            for s in (ranking.Signal(a7, a7_weight), ranking.Signal(bm25, 1 - a7_weight))
            if s.weight > 0
        ]
        return ranking.fuse(signals)

    return mix


def rank_of(scores: Floats, row: int) -> int:
    """Where entity `row` lands in `ranking.top_entities` order (1 = first)."""
    value = scores[row]
    if not np.isfinite(value):
        return int(np.isfinite(scores).sum()) + 1
    ahead = int((scores > value).sum()) + int((scores[:row] == value).sum())
    return ahead + 1


def ranks(signals: Signals, mixer: Mixer) -> NDArray[np.int64]:
    return np.asarray(
        [
            rank_of(mixer(signals.a7[j], signals.bm25[j]), int(signals.right[j]))
            for j in range(len(signals.query_ids))
        ],
        dtype=np.int64,
    )


def hit_at(ranked: NDArray[np.int64], k: int = 10) -> float:
    return float((ranked <= k).mean())


# --- the T7 report ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Subset:
    label: str
    signals: Signals
    """A slice of a set's signals (rows of one language, collection or length)."""


@dataclass(frozen=True, slots=True)
class T7Scores:
    subsets: tuple[Subset, ...]
    grid: dict[float, float]
    """A7 weight -> T7 hit@10 (French, tune)."""
    best_weight: float


def subsets(
    index: WebIndex,
    french: Sequence[Query],
    french_signals: Signals,
    english: Sequence[Query],
    english_signals: Signals,
) -> tuple[Subset, ...]:
    """T7, T7en, T7 on the images T7en covers, by collection, by length."""
    collection = {r.entity_id: r.collection for r in index.catalog.records}
    with_english = {q.query_image_id for q in english}
    out = [
        Subset(SET_ID, french_signals),
        Subset(EN_SET_ID, english_signals),
        Subset(
            f"{SET_ID}-fr-of-en",
            _rows(french_signals, [q.query_image_id in with_english for q in french]),
        ),
    ]
    for name in ("memoire", "joconde", "rose-valland"):
        keep = [collection.get(q.expected_entity_ids[0]) == name for q in french]
        out.append(Subset(f"{SET_ID}-{name}", _rows(french_signals, keep)))
    short = [len(q.text.split()) <= SHORT_WORDS for q in french]
    out.append(Subset(f"{SET_ID}-words", _rows(french_signals, short)))
    out.append(Subset(f"{SET_ID}-sentences", _rows(french_signals, [not s for s in short])))
    return tuple(s for s in out if s.signals.query_ids)


def _rows(signals: Signals, keep: Sequence[bool]) -> Signals:
    rows = np.flatnonzero(np.asarray(keep, dtype=bool))
    return Signals(
        tuple(signals.query_ids[i] for i in rows),
        signals.a7[rows],
        signals.bm25[rows],
        signals.right[rows],
    )


def pick_weight(signals: Signals) -> tuple[dict[float, float], float]:
    """T7 hit@10 for each mixed A7 weight of the grid; the best, ties to the one nearest 0.5.

    A7 alone is not a candidate: without BM25, names ("Corot", "Pont du Gard"), which T7
    leaves out by design, would no longer find their record."""
    grid = {w: hit_at(ranks(signals, weighted(w))) for w in WEIGHT_GRID}
    for w, h in grid.items():
        logger.info("T7 A7 weight %.1f: hit@10 %.3f", w, h)
    best = max(sorted(WEIGHT_GRID, key=lambda w: abs(w - 0.5)), key=lambda w: grid[w])
    return grid, best


def score_options(
    index: WebIndex,
    parts: Sequence[Subset],
    text_run: VectorSet,
    *,
    round_id: str,
    grid: Mapping[float, float],
    best: float,
) -> list[RunResult]:
    """One result file per option: BM25, A7, the current mix, the best mix, RRF."""
    options: list[tuple[str, Mixer, str]] = [
        ("B4-bm25", weighted(0.0), "French BM25 over record text alone"),
        ("B2-A7-text", weighted(1.0), "EmbeddingGemma 2 text vector against images alone"),
        (
            f"B2+B4-mix{CURRENT_A7_WEIGHT:.1f}",
            weighted(CURRENT_A7_WEIGHT),
            "the site's mix before tuning: A7 0.5, BM25 0.5 (min-max, weighted sum)",
        ),
    ]
    if best != CURRENT_A7_WEIGHT:
        options.append(
            (
                f"B2+B4-mix{best:.1f}",
                weighted(best),
                f"the best A7 weight of the grid on T7 tune hit@10: A7 {best:.1f}, "
                f"BM25 {1 - best:.1f}",
            )
        )
    grid_text = ", ".join(f"{w:.1f}: {100 * h:.1f}" for w, h in grid.items())
    out: list[RunResult] = []
    for option, mixer, what in options:
        scores = [_set_score(index, p, mixer) for p in parts]
        info = RunInfo(
            option=option.split("-")[0],
            model=f"{text_run.info.model} text + BM25",
            resolution=text_run.info.resolution,
            pooling="words",
            gallery=GALLERY_LABEL,
            created=_today(),
            cost_usd=text_run.info.cost_usd,
            wall_time_s=text_run.info.wall_time_s,
            route=Route.HOSTED,
            notes=(
                f"{what}. The site's ranking (`engine.word_signals`, `ranking.fuse`) on index "
                f"{WEB_INDEX}; T7 words by {WRITER.model}; A7 text vectors from "
                f"{text_run.info.notes}. T7 hit@10 by A7 weight: {grid_text}"
            ),
        )
        out.append(
            write_result(
                eval_result_path(round_id, "B", option),
                round_id=round_id,
                step="B",
                option=option,
                vector_run=f"{INDEXES[WEB_INDEX].a7_run}+bm25",
                query_vector_run=_text_run_name(text_run),
                vectors=info,
                scores=scores,
                today=_today(),
            )
        )
    return out


def _text_run_name(run: VectorSet) -> str:
    return f"{run.info.option}-{run.info.gallery}"


def _set_score(index: WebIndex, part: Subset, mixer: Mixer) -> SetScore:
    ranked = ranks(part.signals, mixer)
    result = ScoreResult(
        per_query=tuple(
            QueryScore(q, int(r), 1.0 / int(r))
            for q, r in zip(part.signals.query_ids, ranked, strict=True)
        ),
        skipped_no_answer=0,
        gallery_images=len(index.catalog.images),
        gallery_entities=index.catalog.entity_count,
        ap_max_rank=None,
    )
    return set_score(result, test_set=part.label, half="tune", gallery=GALLERY_LABEL)


def text_vectors(run: VectorSet, queries: Sequence[Query]) -> Floats:
    row = run.index()
    missing = [q.query_id for q in queries if q.query_id not in row]
    if missing:
        msg = f"{len(missing)} queries have no text vector, e.g. {missing[:3]}"
        raise ValueError(msg)
    return np.asarray(run.vectors[[row[q.query_id] for q in queries]], dtype=np.float32)


def score_t7(settings: Settings, text_run: str, *, round_id: str) -> T7Scores:
    """Score every option on T7 tune and its subsets; write one result file per option."""
    index = WebIndex.load(settings)
    run = read_vectors(settings.vectors_path(text_run))
    french = load_queries(settings, SET_ID, "tune")
    english = load_queries(settings, EN_SET_ID, "tune")
    fr = word_query_signals(index, french, text_vectors(run, french))
    en = word_query_signals(index, english, text_vectors(run, english))
    parts = subsets(index, french, fr, english, en)
    grid, best = pick_weight(fr)
    score_options(index, parts, run, round_id=round_id, grid=grid, best=best)
    return T7Scores(parts, grid, best)


# --- words refining a photo ------------------------------------------------------------------


def place_queries(settings: Settings, catalog: Catalog) -> list[Query]:
    """T1, T2 and T5 tune photos whose right record has a commune, with it as words."""
    commune: dict[str, str] = {}
    for r in catalog.records:
        if r.commune and r.entity_id not in commune:
            commune[r.entity_id] = r.commune
    out: list[Query] = []
    for set_id in REFINE_SETS:
        for q in load_queries(settings, set_id, "tune"):
            place = commune.get(q.expected_entity_ids[0]) if q.expected_entity_ids else None
            if place:
                out.append(replace(q, text=place))
    return out


def place_id(query: Query) -> str:
    """Id of a refine query's words in the text vector run."""
    return f"{query.query_id}-place"


@dataclass(frozen=True, slots=True)
class RefineScores:
    hits: dict[str, dict[float, float]]
    """Set -> refine weight -> hit@10."""
    queries: dict[str, int]


def score_refine(
    settings: Settings, text_run: str, words: TextWeights, chroma: Mapping[str, float]
) -> RefineScores:
    """Hit@10 of photo + commune for each refine weight, the photo routed as on the site."""
    index = WebIndex.load(settings)
    folder = index.folder
    routing = read_routing(folder / ROUTING_FILE)
    guess = read_guess(folder / GUESS_FILE)
    a3: NDArray[np.float16] = np.load(folder / A3_FILE, mmap_mode="r")
    spec = INDEXES[WEB_INDEX]
    run = read_vectors(settings.vectors_path(text_run))
    row = run.index()
    hits: dict[str, dict[float, float]] = {}
    counts: dict[str, int] = {}
    for set_id in REFINE_SETS:
        queries = [
            q
            for q in place_queries(settings, index.catalog)
            if q.set_id == set_id and q.expected_entity_ids[0] in index.catalog.entity_row
        ]
        if set_id == "T2":
            q7_run = read_vectors(settings.vectors_path(spec.a7_query_run or ""))
            q3_run = read_vectors(settings.vectors_path(spec.a3_query_run or ""))
        else:
            q7_run = q3_run = None
        queries = [
            q
            for q in queries
            if q.query_image_id in (q7_run.index() if q7_run else index.catalog.image_row)
        ]
        q7 = _photo_vectors(queries, q7_run, index.a7, index.catalog)
        q3 = _photo_vectors(queries, q3_run, a3, index.catalog)
        hidden = [(q.query_image_id, *q.exclude_image_ids) for q in queries]
        e7 = entity_cosines(q7, index.a7, index.catalog, hidden)
        e3 = entity_cosines(q3, a3, index.catalog, hidden)
        text = np.asarray(run.vectors[[row[place_id(q)] for q in queries]], dtype=np.float32)
        text_cos = _cosines(text, index.a7)
        per_weight: dict[float, list[bool]] = {w: [] for w in REFINE_GRID}
        for j, q in enumerate(queries):
            photo_type, _ = guess.photo_type(q7[j])
            w = routing.choose(photo_type, chroma.get(q.query_image_id)).a7_weight
            photo = [
                s for s in (ranking.Signal(e7[j], w), ranking.Signal(e3[j], 1 - w)) if s.weight
            ]
            mask = index.hidden_mask(q)
            said = word_signals(index.catalog, text_cos[j], index.bm25.scores(q.text), mask, words)
            right = index.catalog.entity_row[q.expected_entity_ids[0]]
            for refine in REFINE_GRID:
                signals = photo + [ranking.Signal(s.per_entity, s.weight * refine) for s in said]
                fused = ranking.fuse([s for s in signals if s.weight > 0])
                per_weight[refine].append(rank_of(fused, right) <= 10)
        hits[set_id] = {w: float(np.mean(v)) for w, v in per_weight.items()}
        counts[set_id] = len(queries)
        logger.info("refine %s (%d queries): %s", set_id, len(queries), hits[set_id])
    return RefineScores(hits, counts)


def _photo_vectors(
    queries: Sequence[Query],
    run: VectorSet | None,
    gallery: NDArray[np.float16],
    catalog: Catalog,
) -> Floats:
    """Query photo vectors: from the query run (Commons photos), else the index's own row."""
    if run is not None:
        row = run.index()
        return np.asarray(run.vectors[[row[q.query_image_id] for q in queries]], np.float32)
    rows = [catalog.image_row[q.query_image_id] for q in queries]
    return np.asarray(gallery[rows], dtype=np.float32)


def text_inputs(settings: Settings) -> tuple[list[str], list[str]]:
    """Ids and words to embed: T7 and T7en tune queries, and the refine queries' communes."""
    catalog = read_catalog(settings.web_index_dir(WEB_INDEX) / CATALOG_FILE)
    queries = [*load_queries(settings, SET_ID, "tune"), *load_queries(settings, EN_SET_ID, "tune")]
    ids = [q.query_id for q in queries]
    texts = [q.text for q in queries]
    places = place_queries(settings, catalog)
    return ids + [place_id(q) for q in places], texts + [q.text for q in places]


def _today() -> date:
    return datetime.now(UTC).date()
